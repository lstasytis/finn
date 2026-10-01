# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Island flow for Alveo (Vitis): the compute kernel is built with islands, the per-model
v++ link implements the rest.

A Vitis platform keeps the kernels inside its reconfigurable dynamic region (level0_i/ulp),
where Vivado cannot swap a cell of a routed design, so there is no cached shell as on Zynq.
Instead, per model:

  * the compute partition (AXI streams only) is synthesized per node, cut into islands,
    placed and routed in parallel inside ISLAND_REGION and stitched with RapidWright
    (flow.islands_and_stitch) -> stitched core checkpoint (module finn_accel_core);
  * the compute kernel is packaged as a Vitis RTL kernel whose core is a black box, with the
    kernel name and stream arguments FINN's CreateVitisXO would give it;
  * v++ --link runs as in FINN's flow (IODMA kernels, connectivity) with a hook before
    opt_design that reads the stitched core into the black box, locks its placement and keeps
    all other logic out of the island region (EXCLUDE_PLACEMENT). Vivado then implements the
    rest of the dynamic region (IODMAs, HBM subsystem, interconnect), routes the boundary and
    clock nets and writes the xclbin.
"""

import json
import os
import time

from finn.util.dynarapid.components import component_name, vendor_ip_cores
from finn.util.dynarapid.graph import external_ports, kernel_wrapper_verilog
from finn.util.dynarapid.alveo import placeholder_xo_tcl
from finn.util.dynarapid.tools import run_vivado, usable_cpus, vivado_slots
from finn.util.rwislands.device import load_device, pblock_ranges
from finn.util.rwislands.flow import islands_and_stitch, synthesize
from finn.util.rwislands.netlist import TOP_MODULE, channel_graph

# island region per part: rectangles (tile x0, x1, y0, y1), one per SLR (islands do not cross
# SLR boundaries). xcu55c: SLR1 then SLR2 (tile rows 240-479 / 480-719, 5-row margins), left of
# the static base logic (SLICE X4-X170 = tile columns 3-108); SLR0 holds the HBM and the IODMAs
# Rows are whole clock regions (60 tile rows): with the cached platform region the core is a
# reconfigurable partition, whose pblock Vivado snaps to whole clock-region rows (SNAPPING_MODE);
# columns keep a margin inside the partition pblock (core_rp_rect) for the horizontal snapping.
ISLAND_REGION = {"xcu55c-fsvh2892-2L-e": [(6, 105, 240, 479), (6, 105, 480, 719)]}
RP_MARGIN_COLS = 3


def core_rp_rect(region):
    """Pblock of the core partition: the island region's bounding box, a few columns wider."""
    x0, x1, y0, y1 = bounding(region)
    return (max(0, x0 - RP_MARGIN_COLS), x1 + RP_MARGIN_COLS, y0, y1)


def bounding(rects):
    """Bounding rectangle of one or several tile rectangles (the core partition's pblock)."""
    rects = rects if isinstance(rects, list) else [rects]
    return (min(r[0] for r in rects), max(r[1] for r in rects), min(r[2] for r in rects), max(r[3] for r in rects))


def link_hook_tcl(core_dcp, ranges):
    """Vivado hook before opt_design of the v++ link (run.impl_1.STEPS.OPT_DESIGN.TCL.PRE)."""
    return "\n".join(
        [
            "set t0 [clock milliseconds]",
            # the platform's per-IP synthesis uniquifies REF_NAME, ORIG_REF_NAME keeps it
            "set core [get_cells -hier -quiet -filter {IS_BLACKBOX && (REF_NAME =~ *%s* || ORIG_REF_NAME =~ *%s*)}]"
            % (TOP_MODULE, TOP_MODULE),
            'if {[llength $core] != 1} {',
            '  puts "RWI_HOOK black boxes: [get_cells -hier -quiet -filter IS_BLACKBOX]"',
            '  error "island core black box not found: $core"',
            "}",
            # read_checkpoint -cell replaces the cell object: keep its name, query it again
            "set core_name [get_property NAME $core]",
            "read_checkpoint -cell $core_name %s" % core_dcp,
            'puts "RWI_HOOK read_checkpoint [expr ([clock milliseconds] - $t0) / 1000.0]"',
            "set core [get_cells $core_name]",
            # placement locked; its routing stays, Vivado may still finish/repair nets
            "lock_design -level placement $core",
            # the island region is the core's alone (one resize_pblock call: 2024.2 drops sites
            # when a pblock is grown in small steps). No other pblocks: v++'s own
            # pblock_dynamic_SLR<n> must stay as they are (HDPR-23, VPL 30-887)
            "create_pblock pblock_islands",
            "resize_pblock pblock_islands -add {%s}" % " ".join(ranges),
            "add_cells_to_pblock pblock_islands $core",
            "set_property EXCLUDE_PLACEMENT 1 [get_pblocks pblock_islands]",
            'puts "RWI_HOOK done $core [expr ([clock milliseconds] - $t0) / 1000.0]"',
        ]
    ) + "\n"


def rw_islands_kernel(
    kernel_model, kernel_name, part, clk_ns, out_dir, islands="auto", workers=None, subdivide=False
):
    """Build the compute kernel with the island flow. kernel_model: a compute partition (AXI
    streams only) whose nodes carry generated IP. Returns a result dict with status, the stitched
    core checkpoint (core_dcp), the black-box kernel .xo (xo) and the v++ link hook (hook)."""
    os.makedirs(out_dir, exist_ok=True)
    cpus = workers or usable_cpus()
    _, slots = vivado_slots()
    work = os.path.join(out_dir, "work")
    res = {"out_dir": out_dir, "kernel": kernel_name, "part": part, "clk_ns": clk_ns}
    stamps = {}
    t_total = time.time()

    def stamp(k):
        stamps[k] = time.time() - t_total

    def done(status, **kw):
        res.update(status=status, **kw)
        res["stamps"] = stamps
        res["total_s"] = time.time() - t_total
        with open(os.path.join(out_dir, "rwislands_kernel.json"), "w") as f:
            json.dump(res, f, indent=2)
        return res

    km = kernel_model
    km.save(os.path.join(out_dir, "kernel.onnx"))
    vendor = {n.name: vendor_ip_cores(n) for n in km.graph.node}
    vendor = {k: v for k, v in vendor.items() if v}
    if vendor:
        return done("unsupported_vendor_ip", vendor_ip=vendor)
    g = channel_graph(km)
    dcps = {n.name: component_name(km, n, part, clk_ns) for n in km.graph.node}
    dev = load_device(part)

    # black-box kernel .xo, independent of the implementation: meanwhile in a thread would
    # overlap it, but it is short (~1 min) next to synthesis; packaged first
    ins, outs = external_ports(km)
    xo_dir = os.path.join(out_dir, "xo")
    os.makedirs(xo_dir, exist_ok=True)
    wrapper = os.path.join(xo_dir, kernel_name + ".v")
    with open(wrapper, "w") as f:
        f.write(kernel_wrapper_verilog(kernel_name, TOP_MODULE, ins, outs, black_box=True))
    tcl, xo = placeholder_xo_tcl(kernel_name, ins, outs, [wrapper], part, xo_dir)
    with open(os.path.join(xo_dir, "package.tcl"), "w") as f:
        f.write(tcl)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as ex:
        f_xo = ex.submit(
            run_vivado, os.path.join(xo_dir, "package.tcl"), os.path.join(xo_dir, "package.log"), xo_dir
        )
        synth = synthesize(km, dcps, work, part, clk_ns, cpus, slots)
        stamp("synth")
        f_xo.result()
    res["synth"] = synth
    if not os.path.isfile(xo):
        return done("xo_failed")
    failed = [d for d, r in synth.items() if r["status"] != "ok"]
    if failed:
        return done("synth_failed", failed=failed)

    region = ISLAND_REGION[part]
    fail, core_dcp = islands_and_stitch(
        km, g, dcps, synth, dev, part, clk_ns, work, cpus, slots, islands, region, [], res, stamp
    )
    if fail is not None:
        return done(fail)
    # reserve only the islands' rectangles (not the whole island region: excluding most of an
    # SLR made the placement of the rest of the dynamic region slower, 28 vs 22 min on TFC)
    ranges = []
    for info in res["islands"].values():
        for r in info["rects"]:
            ranges += pblock_ranges(dev.sites_in(*r))
    hook = os.path.join(out_dir, "link_hook.tcl")
    with open(hook, "w") as f:
        if subdivide:
            # shell-building link: the core partition gets the whole island region
            f.write(
                subdivide_hook_tcl(
                    core_dcp, pblock_ranges(dev.sites_in(*core_rp_rect(region))), os.path.join(out_dir, "ulp_bb.dcp")
                )
            )
        else:
            f.write(link_hook_tcl(core_dcp, ranges))
    return done("ok", core_dcp=core_dcp, xo=xo, hook=hook)


# ---------------------------------------------------------------------------------------------
# Cached platform region (nested DFX). The first link of a kernel interface subdivides the
# platform's reconfigurable dynamic region (level0_i/ulp) with pr_subdivide so that the compute
# core becomes a reconfigurable partition of its own; the routed result with the core emptied is
# cached ("shell"). Later models with the same interface only fill the core partition, route its
# boundary and write the dynamic region's partial bitstream into a copy of the shell's xclbin.
# ---------------------------------------------------------------------------------------------

ULP_CELL = "level0_i/ulp"


def subdivide_hook_tcl(core_dcp, region_ranges, ulp_dcp):
    """Hook before opt_design of the shell-building v++ link: the core black box becomes a
    reconfigurable partition (pr_subdivide of the dynamic region, from the linked netlist saved
    to ulp_dcp), gets a pblock over the whole island region (so that later models fit), and is
    filled with this model's stitched core (placement locked)."""
    return "\n".join(
        [
            "set t0 [clock milliseconds]",
            'proc rwi_stamp {n} {global t0; puts "RWI_HOOK $n [expr ([clock milliseconds] - $t0) / 1000.0]"}',
            "set core [get_cells -hier -quiet -filter {IS_BLACKBOX && (REF_NAME =~ *%s* || ORIG_REF_NAME =~ *%s*)}]"
            % (TOP_MODULE, TOP_MODULE),
            'if {[llength $core] != 1} {error "island core black box not found: $core"}',
            "set core_name [get_property NAME $core]",
            # the linked dynamic region with the core as a black box = its new static logic
            "write_checkpoint -force -cell %s %s" % (ULP_CELL, ulp_dcp),
            "rwi_stamp ulp_saved",
            "update_design -cell %s -black_box" % ULP_CELL,
            "pr_subdivide -cell %s -subcells [list $core_name] %s" % (ULP_CELL, ulp_dcp),
            "rwi_stamp pr_subdivide",
            'puts "RWI_HOOK partitions: [get_cells -hier -quiet -filter {HD.RECONFIGURABLE}]"',
            "create_pblock pblock_core_rp",
            "resize_pblock pblock_core_rp -add {%s}" % " ".join(region_ranges),
            "add_cells_to_pblock pblock_core_rp [get_cells $core_name]",
            "set_property SNAPPING_MODE ON [get_pblocks pblock_core_rp]",
            "read_checkpoint -cell $core_name %s" % core_dcp,
            "rwi_stamp read_core",
            "lock_design -level placement [get_cells $core_name]",
            'puts "RWI_HOOK done $core_name [expr ([clock milliseconds] - $t0) / 1000.0]"',
        ]
    ) + "\n"


def shell_key(platform, clk_ns, signature):
    import hashlib

    from finn.util.dynarapid.tools import vivado_version

    sig = {"platform": platform, "clk_ns": clk_ns, "kernels": signature, "version": 1, "vivado": vivado_version()}
    return "ushell" + hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]


def partition_signature(kernel_model, name):
    """What the cached shell depends on: IODMA kernels' parameters and the compute kernel's
    stream interface (kernel name, widths)."""
    from qonnx.custom_op.registry import getCustomOp

    nodes = []
    for n in kernel_model.graph.node:
        if n.op_type.startswith("IODMA"):
            inst = getCustomOp(n)
            nodes.append(
                {
                    k: inst.get_nodeattr(k)
                    for k in ("intfWidth", "streamWidth", "direction", "burstMode", "NumChannels")
                    if k in inst.get_nodeattr_types()
                }
            )
    if nodes:
        return {"name": name, "iodma": nodes}
    ins, outs = external_ports(kernel_model)
    return {"name": name, "ins": [p["width"] for p in ins], "outs": [p["width"] for p in outs]}


def extract_shell(link_dir, shell_dir, info):
    """After the shell-building link: routed design with the core partition emptied, all routing
    locked, plus the xclbin (its metadata stays valid for every core with this interface)."""
    import glob
    import shutil

    impl = os.path.join(link_dir, "_x/link/vivado/vpl/prj/prj.runs/impl_1")
    routed = glob.glob(os.path.join(impl, "*_routed.dcp"))
    xclbin = os.path.join(link_dir, "a.xclbin")
    if not routed or not os.path.isfile(xclbin):
        return {"status": "shell_extract_failed", "reason": "no routed dcp / xclbin"}
    work = shell_dir + ".work"
    os.makedirs(work, exist_ok=True)
    tcl = os.path.join(work, "extract.tcl")
    with open(tcl, "w") as f:
        f.write(
            "\n".join(
                [
                    "open_checkpoint %s" % routed[0],
                    "set core [get_cells -hier -filter {HD.RECONFIGURABLE}]",
                    'if {[llength $core] != 1} {error "core partition not found: $core"}',
                    "set f [open %s/core_cell.txt w]; puts $f [get_property NAME $core]; close $f" % work,
                    "update_design -cell $core -black_box",
                    "lock_design -level routing",
                    # the core's static pins will join the VCC/GND nets: keep those changeable
                    # (done once here, not per model: a hierarchical net query on the platform)
                    "set_property IS_ROUTE_FIXED 0 [get_nets -hier -quiet -filter {TYPE == POWER || TYPE == GROUND}]",
                    "write_checkpoint -force %s/shell_routed.dcp" % work,
                ]
            )
            + "\n"
        )
    rc, t = run_vivado(tcl, os.path.join(work, "extract.log"), work)
    if rc != 0 or not os.path.isfile(os.path.join(work, "shell_routed.dcp")):
        return {"status": "shell_extract_failed", "log": os.path.join(work, "extract.log")}
    shutil.copy(xclbin, os.path.join(work, "shell.xclbin"))
    res = dict(info, status="built", extract_s=t, core_cell=open(os.path.join(work, "core_cell.txt")).read().strip())
    with open(os.path.join(work, "shell.json"), "w") as f:
        json.dump(res, f, indent=2)
    if os.path.isdir(shell_dir):
        shutil.rmtree(shell_dir)
    os.rename(work, shell_dir)
    return res


def assemble_cached(shell_dir, core_dcp, out_dir, threads=16):
    """Fill the cached shell's core partition, route, write the dynamic region's partial
    bitstream and package it into a copy of the shell's xclbin. Returns a result dict."""
    import re
    import shutil
    import subprocess

    shell = json.load(open(os.path.join(shell_dir, "shell.json")))
    os.makedirs(out_dir, exist_ok=True)
    bit = os.path.join(out_dir, "partial.bit")
    tcl = os.path.join(out_dir, "assemble.tcl")
    with open(tcl, "w") as f:
        f.write(
            "\n".join(
                [
                    "set_param general.maxThreads %d" % threads,
                    "set t0 [clock milliseconds]",
                    'proc stamp {n} {global t0; puts "STAMP $n [expr ([clock milliseconds] - $t0) / 1000.0]"}',
                    "open_checkpoint %s/shell_routed.dcp" % shell_dir,
                    "stamp open_shell",
                    "read_checkpoint -cell %s %s" % (shell["core_cell"], core_dcp),
                    "stamp read_core",
                    "route_design",
                    "stamp route",
                    "report_route_status -file %s/route_status.rpt" % out_dir,
                    "report_timing_summary -file %s/timing_summary.rpt" % out_dir,
                    "stamp reports",
                    "write_bitstream -force -cell %s %s" % (ULP_CELL, bit),
                    "stamp bitstream",
                    "report_utilization -hierarchical -hierarchical_depth 6 -format xml -file %s/synth_report.xml"
                    % out_dir,
                ]
            )
            + "\n"
        )
    log = os.path.join(out_dir, "assemble.log")
    rc, t = run_vivado(tcl, log, out_dir)
    res = {"assembly_s": t, "rc": rc}
    txt = open(log, errors="ignore").read()
    res["stamps"] = {k: float(v) for k, v in re.findall(r"^STAMP (\w+) ([\d.]+)", txt, re.M)}
    rs = os.path.join(out_dir, "route_status.rpt")
    if os.path.isfile(rs):
        m = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", open(rs).read())
        res["routing_errors"] = int(m.group(1)) if m else None
    if rc != 0 or not os.path.isfile(bit):
        res["status"] = "assembly_failed"
        return res
    xclbin = os.path.join(out_dir, "finn-accel.xclbin")
    p = subprocess.run(
        ["xclbinutil", "--input", os.path.join(shell_dir, "shell.xclbin"), "--replace-section",
         "BITSTREAM:RAW:%s" % bit, "--force", "--output", xclbin],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    open(os.path.join(out_dir, "xclbinutil.log"), "w").write(p.stdout)
    res["status"] = "ok" if p.returncode == 0 and os.path.isfile(xclbin) else "xclbin_failed"
    res["xclbin"] = xclbin
    res["synth_report"] = os.path.join(out_dir, "synth_report.xml")
    return res
