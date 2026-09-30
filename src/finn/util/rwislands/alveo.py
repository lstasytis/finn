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

# island region (tile x0, x1, y0, y1) per part: xcu55c SLR1 (tile rows 240-479, 5-row margins),
# left of the static base logic (SLICE X4-X170 = tile columns 3-108)
ISLAND_REGION = {"xcu55c-fsvh2892-2L-e": (3, 108, 245, 474)}


def link_hook_tcl(core_dcp, ranges):
    """Vivado hook before opt_design of the v++ link (run.impl_1.STEPS.OPT_DESIGN.TCL.PRE)."""
    return "\n".join(
        [
            "set t0 [clock milliseconds]",
            # the platform's per-IP synthesis uniquifies REF_NAME, ORIG_REF_NAME keeps it
            "set core [get_cells -hier -filter {ORIG_REF_NAME == %s || REF_NAME == %s}]"
            % (TOP_MODULE, TOP_MODULE),
            'if {[llength $core] != 1} {error "island core black box not found: $core"}',
            "read_checkpoint -cell $core %s" % core_dcp,
            'puts "RWI_HOOK read_checkpoint [expr ([clock milliseconds] - $t0) / 1000.0]"',
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


def rw_islands_kernel(kernel_model, kernel_name, part, clk_ns, out_dir, islands="auto", workers=None):
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
    hook = os.path.join(out_dir, "link_hook.tcl")
    with open(hook, "w") as f:
        f.write(link_hook_tcl(core_dcp, pblock_ranges(dev.sites_in(*region))))
    return done("ok", core_dcp=core_dcp, xo=xo, hook=hook)
