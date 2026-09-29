# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-implemented Vitis (Alveo) shell for the DynaRapid bitfile flow.

FINN's Vitis flow links three kernels per model (alveo_build.VitisLink): idma0 (HLS IODMA,
m_axi + s_axilite), the compute partition (RTL kernel, AXI-Stream only) and odma0, and v++
implements the platform's dynamic region (ULP, a DFX partition) around them on top of the
locked static region. The ULP also contains encrypted IP (HBM memory subsystem, SmartConnect),
so it stays in Vivado. As in the Zynq shell flow (shell.py), the part that does not depend on
the model is implemented once and cached:

  * shell: v++ --link of the real IODMA kernels with a placeholder compute kernel of the same
    name and stream widths. The placeholder core (registered, DONT_TOUCH, see
    shell.core_verilog) is constrained to the DynaRapid region; every other ULP cell is kept
    out of it and its routing contained (pblocks, applied by a pre-placement hook), so the
    region is free for the DynaRapid-routed compute kernel. Cached per platform, clock, kernel
    names and stream widths;
  * per model: DynaRapid places and routes the compute kernel within that region;
  * assembly (assemble_tcl): open the shell's routed checkpoint, turn the placeholder core
    into a black box, read_checkpoint -cell the routed compute kernel, route the boundary
    nets, write the ULP partial bitstream and put it into a copy of the shell's xclbin
    (xclbinutil), whose metadata (kernels, connectivity, memory topology) is unchanged.
"""

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from finn.util.dynarapid.components import vendor_ip_cores
from finn.util.dynarapid.flow import build_library, dynarapid_pnr
from finn.util.dynarapid.graph import external_ports, kernel_wrapper_verilog
from finn.util.dynarapid.shell import CORE_MODULE, core_verilog
from finn.util.dynarapid.tools import run_vivado, vivado_version

# the ULP partition of the Alveo platforms (xilinx_*_gen3x16_xdma_*)
ULP_CELL = "level0_i/ulp"

# DynaRapid region per part, as SLICE ranges (Vivado pblock syntax) and as DynaRapid map
# coordinates (starti,startj,endi,endj; map row 0 is the top of the device).
# xcu55c: SLR2 + SLR1 (SLICE Y252-707, map rows 12-467; map row r = SLICE row 719 - r), left
# of the static base logic (pblock_blp, X197-232 / X206-232 in SLR2). SLR1 alone (216 rows) is
# too short: DSP/BRAM components relocate only vertically, and the VGG10 MVAU variants all sit
# on two column bands.
DR_REGION = {
    "xcu55c-fsvh2892-2L-e": {
        # corner slices (top-left, bottom-right); map columns 3-108 = SLICE X4-X170
        "corners": ("SLICE_X4Y707", "SLICE_X170Y252"),
        # DynaRapid place region "top,bottom,left,right" (map rows / columns)
        "place": "12,467,3,108",
    },
}


def shell_key(platform, clk_ns, kernels, region=None):
    """kernels: [{"name", "xo" (None for the compute kernel), "ins", "outs"}], in link order."""
    sig = {
        "platform": platform,
        "clk_ns": clk_ns,
        "kernels": [
            {
                "name": k["name"],
                "ins": [c["width"] for c in k.get("ins", [])],
                "outs": [c["width"] for c in k.get("outs", [])],
                # the IODMA kernels only depend on their widths, the xo content is FINN's
                "dma": k["xo"] is not None,
            }
            for k in kernels
        ],
        "version": 1,
        "vivado": vivado_version(),
        "region": region,
    }
    h = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]
    return "vshell%s" % h


def placeholder_sources(kernel_name, ins, outs, out_dir):
    """Wrapper (the compute kernel's AXI-Stream interface) around the placeholder core."""
    os.makedirs(out_dir, exist_ok=True)
    wrapper = os.path.join(out_dir, kernel_name + ".v")
    with open(wrapper, "w") as f:
        f.write(kernel_wrapper_verilog(kernel_name, CORE_MODULE, ins, outs, black_box=False))
    core = os.path.join(out_dir, CORE_MODULE + ".v")
    with open(core, "w") as f:
        f.write(core_verilog([{"in": ins, "out": outs}]))
    return [wrapper, core]


def placeholder_xo_tcl(kernel_name, ins, outs, srcs, part, out_dir):
    """Package the placeholder as a Vitis RTL kernel, with the kernel arguments that
    alveo_build.CreateVitisXO gives the real compute kernel (streams only)."""
    args = []
    arg_id = 0
    for k, c in enumerate(ins):
        args.append("{s_axis_%d:4:%d:s_axis_%d:0x0:0x0:ap_uint&lt;%d>:0}" % (k, arg_id, k, c["width"]))
        arg_id += 1
    for k, c in enumerate(outs):
        args.append("{m_axis_%d:4:%d:m_axis_%d:0x0:0x0:ap_uint&lt;%d>:0}" % (k, arg_id, k, c["width"]))
        arg_id += 1
    xo = os.path.join(out_dir, kernel_name + ".xo")
    tcl = [
        "create_project -force placeholder %s/proj -part %s" % (out_dir, part),
        "add_files {%s}" % " ".join(srcs),
        "set_property top %s [current_fileset]" % kernel_name,
        "update_compile_order -fileset sources_1",
        "ipx::package_project -root_dir %s/ip -vendor xilinx.com -library RTLKernel "
        "-taxonomy /KernelIP -import_files -set_current true" % out_dir,
        "set core [ipx::current_core]",
        "set_property sdx_kernel true $core",
        "set_property sdx_kernel_type rtl $core",
        "set_property supported_families { } $core",
        # as CreateStitchedIP: without it v++ rejects the IP for the part (VPL 5-683)
        "set_property auto_family_support_level level_2 $core",
        "ipx::update_checksums $core",
        "ipx::save_core $core",
        "close_project",
        "package_xo -force -xo_path %s -kernel_name %s -ip_directory %s/ip %s"
        % (xo, kernel_name, out_dir, " ".join("-kernel_xml_args " + a for a in args)),
    ]
    return "\n".join(tcl) + "\n", xo


def pre_place_tcl(compute_inst, corners):
    """Pre-placement hook of the shell link: the placeholder core goes into the DynaRapid
    region (EXCLUDE_PLACEMENT, CONTAIN_ROUTING), and all other ULP cells into the rest of
    the dynamic region with contained routing, so their routes stay out of it."""
    return "\n".join(
        [
            # the platform's per-IP synthesis uniquifies REF_NAME, ORIG_REF_NAME keeps it
            "set core [get_cells -hier -filter {NAME =~ */%s/* && ORIG_REF_NAME == %s}]"
            % (compute_inst, CORE_MODULE),
            'if {[llength $core] != 1} {error "DynaRapid placeholder core not found: $core"}',
            # all SLICE / DSP / RAMB sites of the tiles in the corners' grid rectangle, in one
            # resize_pblock call (Vivado 2024.2 drops sites when adding in small chunks)
            "set t0 [get_tiles -of [get_sites %s]]" % corners[0],
            "set t1 [get_tiles -of [get_sites %s]]" % corners[1],
            "set r0 [get_property ROW $t0]; set r1 [get_property ROW $t1]",
            "set c0 [get_property COLUMN $t0]; set c1 [get_property COLUMN $t1]",
            "set tiles [get_tiles -filter \"ROW >= [expr min($r0,$r1)] && ROW <= [expr max($r0,$r1)] "
            "&& COLUMN >= [expr min($c0,$c1)] && COLUMN <= [expr max($c0,$c1)]\"]",
            "set sites [get_sites -quiet -of $tiles -filter {SITE_TYPE =~ SLICE* || SITE_TYPE =~ DSP48* "
            "|| SITE_TYPE =~ RAMB*}]",
            "create_pblock pblock_dynarapid",
            "resize_pblock pblock_dynarapid -add $sites",
            "add_cells_to_pblock pblock_dynarapid $core",
            "set_property EXCLUDE_PLACEMENT 1 [get_pblocks pblock_dynarapid]",
            "set_property CONTAIN_ROUTING 1 [get_pblocks pblock_dynarapid]",
            "set dyn [get_pblocks pblock_dynamic_region]",
            # v++'s per-SLR child pblocks (pblock_dynamic_SLR<n>) stay as they are: resizing
            # them breaks the clock-region-column rule of reconfigurable pblocks (VPL 30-887);
            # EXCLUDE_PLACEMENT already keeps their cells out of the region
            # the cells placed directly in the dynamic region (kernels, interconnect, HBM
            # subsystem) cannot get a pblock of their own: Vivado then re-parents v++'s SLR
            # pblocks under it (DRC HDPR-23). EXCLUDE_PLACEMENT keeps them out of the region;
            # routes of theirs that pass through it are rerouted by the assembly
            "set rest [get_cells -quiet -of $dyn]",
            'puts "DYNARAPID_REGION core=$core sites=[llength $sites] rest_cells=[llength $rest]"',
        ]
    ) + "\n"


def link_config(kernels, platform_mem="HBM[0]"):
    """v++ connectivity of the FINN Vitis flow (VitisLink): idma -> compute -> odma."""
    cfg = ["[connectivity]"]
    for k in kernels:
        cfg.append("nk=%s:1:%s" % (k["name"], k["inst"]))
        if k.get("mm"):
            cfg.append("sp=%s.m_axi_gmem0:%s" % (k["inst"], platform_mem))
    for a, b in zip(kernels, kernels[1:]):
        cfg.append("stream_connect=%s.m_axis_0:%s.s_axis_0" % (a["inst"], b["inst"]))
    return "\n".join(cfg) + "\n"


def build_shell(platform, part, clk_ns, kernels, shell_lib, jobs=8):
    """Build (or reuse) the Vitis shell. kernels: link order, each {"name", "inst", "xo" or
    None (compute kernel -> placeholder), "ins", "outs", "mm"}. Returns a result dict with
    shell_dir, the routed checkpoint, the xclbin and the placeholder core cell."""
    key = shell_key(platform, clk_ns, kernels, DR_REGION[part])
    shell_dir = os.path.join(shell_lib, key)
    done = os.path.join(shell_dir, "shell.json")
    if os.path.isfile(done):
        res = json.load(open(done))
        res["status"] = "cached"
        return res
    work = shell_dir + ".work"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    xos = []
    compute = None
    for k in kernels:
        if k["xo"] is not None:
            xos.append(k["xo"])
            continue
        compute = k
        pdir = os.path.join(work, "placeholder")
        srcs = placeholder_sources(k["name"], k["ins"], k["outs"], pdir)
        tcl, xo = placeholder_xo_tcl(k["name"], k["ins"], k["outs"], srcs, part, pdir)
        with open(os.path.join(pdir, "package.tcl"), "w") as f:
            f.write(tcl)
        run_vivado(os.path.join(pdir, "package.tcl"), os.path.join(pdir, "package.log"), pdir)
        assert os.path.isfile(xo), "placeholder kernel not packaged, see %s/package.log" % pdir
        xos.append(xo)
    assert compute is not None, "no compute kernel in the shell kernel list"
    with open(os.path.join(work, "config.txt"), "w") as f:
        f.write(link_config(kernels))
        f.write("[vivado]\n")
        f.write("prop=run.impl_1.STEPS.PLACE_DESIGN.TCL.PRE=%s/pre_place.tcl\n" % work)
        f.write("impl.jobs=%d\n" % jobs)
        f.write("synth.jobs=%d\n" % jobs)
    with open(os.path.join(work, "pre_place.tcl"), "w") as f:
        f.write(pre_place_tcl(compute["inst"], DR_REGION[part]["corners"]))
    cmd = [
        "v++", "-t", "hw", "--platform", platform, "--link", *xos,
        "--kernel_frequency", str(round(1000 / clk_ns)), "--config", "config.txt",
        "--optimize", "0", "--save-temps", "-R2", "-o", "shell.xclbin",
        # the platform IP synthesis (138 OOC runs on the U55C) is the same for every shell
        "--remote_ip_cache", os.path.join(shell_lib, "ip_cache"),
    ]
    with open(os.path.join(work, "link.log"), "w") as log:
        rc = subprocess.run(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT).returncode
    xclbin = os.path.join(work, "shell.xclbin")
    routed = glob.glob(os.path.join(work, "_x/link/vivado/vpl/prj/prj.runs/impl_1/*_routed.dcp"))
    if rc != 0 or not os.path.isfile(xclbin) or not routed:
        return {"status": "shell_failed", "shell_dir": work, "rc": rc}
    os.makedirs(shell_dir, exist_ok=True)
    shutil.copy(xclbin, os.path.join(shell_dir, "shell.xclbin"))
    shutil.copy(routed[0], os.path.join(shell_dir, "shell_routed.dcp"))
    res = {
        "status": "built",
        "shell": key,
        "shell_dir": shell_dir,
        "routed_dcp": os.path.join(shell_dir, "shell_routed.dcp"),
        "xclbin": os.path.join(shell_dir, "shell.xclbin"),
        "compute_kernel": compute["name"],
        "compute_inst": compute["inst"],
        "region": DR_REGION[part],
        "work": work,
    }
    with open(done, "w") as f:
        json.dump(res, f, indent=2)
    return res


def assemble_tcl(shell, accel_dcp, out_dir, bitfile, threads=16):
    """Fill the placeholder core with the DynaRapid-routed compute kernel, route, and write
    the partial bitstream of the ULP."""
    return "\n".join(
        [
            "set_param general.maxThreads %d" % threads,
            "set t0 [clock milliseconds]",
            'proc stamp {name} {global t0; puts "STAMP $name [expr ([clock milliseconds] - $t0) / 1000.0]"}',
            "open_checkpoint %s" % shell["routed_dcp"],
            "stamp open_shell",
            "set core [get_cells -hier -filter {NAME =~ */%s/* && ORIG_REF_NAME == %s}]"
            % (shell["compute_inst"], CORE_MODULE),
            'if {[llength $core] != 1} {error "placeholder core not found: $core"}',
            "update_design -cells $core -black_box",
            "read_checkpoint -cell $core %s" % accel_dcp,
            "stamp read_accel",
            "set_property IS_ROUTE_FIXED 0 [get_nets -hier -quiet -filter {TYPE == POWER || TYPE == GROUND}]",
            "route_design",
            "stamp route",
            "report_route_status -file %s/route_status.rpt" % out_dir,
            "report_timing_summary -file %s/timing_summary.rpt" % out_dir,
            "stamp reports",
            "write_bitstream -force -cell %s %s" % (ULP_CELL, bitfile),
            "stamp bitstream",
        ]
    ) + "\n"


def package_xclbin(shell, partial_bit, xclbin):
    """Copy of the shell's xclbin with the partial bitstream replaced."""
    cmd = [
        "xclbinutil", "--input", shell["xclbin"], "--replace-section",
        "BITSTREAM:RAW:%s" % partial_bit, "--force", "--output", xclbin,
    ]
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _done(res, out_dir, t_total):
    res["total_s"] = time.time() - t_total
    with open(os.path.join(out_dir, "dynarapid_alveo.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def dynarapid_alveo_build(
    compute_model,
    kernels,
    platform,
    part,
    clk_ns,
    out_dir,
    library_dir,
    shell_lib=None,
    workers=None,
):
    """xclbin of a FINN Vitis design whose compute kernel is placed and routed by DynaRapid.

    compute_model: the compute partition (generated IP), kernels: the link order as for
    build_shell, with {"xo": None} for the compute kernel (ins / outs are filled in here).
    Returns a result dict with the xclbin and per-stage times."""
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or os.cpu_count()
    shell_lib = shell_lib or os.path.join(os.path.dirname(os.path.abspath(library_dir)), "shells")
    res = {"out_dir": out_dir, "platform": platform, "part": part, "clk_ns": clk_ns}
    t_total = time.time()
    compute_model.save(os.path.join(out_dir, "compute.onnx"))
    vendor = {n.name: vendor_ip_cores(n) for n in compute_model.graph.node}
    vendor = {k: v for k, v in vendor.items() if v}
    if vendor:
        res["status"] = "unsupported_vendor_ip"
        res["vendor_ip"] = vendor
        return _done(res, out_dir, t_total)
    ins, outs = external_ports(compute_model)
    kernels = [dict(k) for k in kernels]
    for k in kernels:
        if k["xo"] is None:
            k["ins"], k["outs"] = ins, outs
    work = os.path.join(os.path.dirname(os.path.abspath(library_dir)), "work")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_shell = ex.submit(
            build_shell, platform, part, clk_ns, kernels, shell_lib, max(1, workers // 4)
        )
        # the component library does not depend on the shell: build it meanwhile
        f_lib = ex.submit(build_library, compute_model, work, library_dir, part, clk_ns, workers)
        shell = f_shell.result()
        f_lib.result()
    res["parallel_s"] = time.time() - t0
    res["shell"] = shell
    if shell["status"] not in ("built", "cached"):
        res["status"] = "shell_failed"
        return _done(res, out_dir, t_total)

    def accel_job(place_order=None):
        return dynarapid_pnr(
            compute_model,
            os.path.join(out_dir, "dynarapid"),
            library_dir,
            part,
            clk_ns,
            workers=workers,
            graph_name="accel",
            check=False,
            no_clock=True,
            place_region=DR_REGION[part]["place"],
            place_order=place_order,
            rwroute_max_iter=30,
        )

    t0 = time.time()
    accel_res = accel_job()
    log = os.path.join(out_dir, "dynarapid", "generate_design.log")
    if accel_res["status"] == "dynarapid_failed" and "Could not find placement" in open(log).read():
        res["place_retry"] = "size"
        accel_res = accel_job(place_order="size")
    res["stitch_s"] = time.time() - t0
    res["accel"] = {k: v for k, v in accel_res.items() if k != "components"}
    res["accel_components"] = accel_res.get("components")
    if accel_res["status"] != "routed":
        res["status"] = "accel_" + accel_res["status"]
        return _done(res, out_dir, t_total)

    asm_dir = os.path.join(out_dir, "assembly")
    os.makedirs(asm_dir, exist_ok=True)
    partial = os.path.join(asm_dir, "ulp_partial.bit")
    tcl = os.path.join(asm_dir, "assemble.tcl")
    with open(tcl, "w") as f:
        f.write(assemble_tcl(shell, accel_res["routed_dcp"], asm_dir, partial))
    log = os.path.join(asm_dir, "assemble.log")
    rc, res["assembly_s"] = run_vivado(tcl, log, asm_dir)
    txt = open(log, errors="ignore").read()
    res["assembly_stamps"] = {k: float(v) for k, v in re.findall(r"^STAMP (\w+) ([\d.]+)", txt, re.M)}
    m = re.search(r"nets with routing errors[.]*\s*:\s*(\d+)",
                  open(os.path.join(asm_dir, "route_status.rpt")).read()) if os.path.isfile(
        os.path.join(asm_dir, "route_status.rpt")) else None
    res["nets_with_routing_errors"] = int(m.group(1)) if m else None
    res["timing_rpt"] = os.path.join(asm_dir, "timing_summary.rpt")
    partials = [partial] + glob.glob(os.path.join(asm_dir, "*partial*.bit"))
    partials = [p for p in partials if os.path.isfile(p)]
    if rc != 0 or not partials:
        res["status"] = "assembly_failed"
        return _done(res, out_dir, t_total)
    xclbin = os.path.join(out_dir, "finn-accel.xclbin")
    p = package_xclbin(shell, partials[0], xclbin)
    open(os.path.join(asm_dir, "xclbinutil.log"), "w").write(p.stdout)
    if p.returncode != 0 or not os.path.isfile(xclbin):
        res["status"] = "xclbin_failed"
        return _done(res, out_dir, t_total)
    res.update(status="ok", xclbin=xclbin, partial_bit=partials[0])
    return _done(res, out_dir, t_total)
