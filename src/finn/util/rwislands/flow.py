# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Island bitfile flow for Zynq: parallel out-of-context place and route of FINN node groups
at their final location, stitched with RapidWright, inserted into a pre-implemented shell.

    ZynqBuild partitions (idma, kernel, odma) -> one accelerator graph
      -> in parallel: shell (cached per board / clock / IODMA interfaces)
                      out-of-context synthesis of every node      (resource numbers)
      -> islands: the node chain cut into K groups, floorplan (rwislands.floorplan)
      -> in parallel: place and route of every island in its pblock (one Vivado run each)
                      synthesis of the accelerator top (islands as black boxes)
      -> RapidWright: fill the black boxes with the routed islands, route the nets between
                      islands inside the island region (IslandStitcher.java, RWRoute)
      -> Vivado: shell + accelerator, route the remaining (boundary, clock, static) nets,
                 bitstream (rwislands.assembly)

No component library, relocation or placement database: every island is placed and routed
once, where it ends up.
"""

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from finn.util.dynarapid.components import (
    component_name,
    direct_sources,
    direct_synth_tcl,
    synth_tcl,
    vendor_ip_cores,
)
from finn.util.dynarapid.graph import mm_ports
from finn.util.dynarapid.shell import PS_BOUNDARY_INT_X, build_shell
from finn.util.dynarapid.tools import java_bin, run_vivado, usable_cpus, vivado_slots
from finn.util.dynarapid.zynq import _reports, merge_partitions
from finn.util.rwislands.assembly import assemble_tcl
from finn.util.rwislands.device import load_device, pblock_ranges
from finn.util.rwislands.floorplan import floorplan, island_cost, partition
from finn.util.rwislands.netlist import (
    TOP_MODULE,
    boundary_ports,
    channel_graph,
    island_verilog,
    top_verilog,
)
from finn.util.rwislands.profiles import profile, synth_args

# island region of the Vivado-only Alveo shell (dynarapid.shell.ALVEO_SHELL; the shell takes
# clock regions X6-X7 of SLR0 around the PCIe block, the islands one rectangle per SLR; tile
# columns 131-147 (X7) hold transceivers/IO in SLR1/SLR2)
VIVADO_ALVEO_REGION = {
    "xcu55c-fsvh2892-2L-e": [(6, 108, 0, 239), (6, 130, 240, 479), (6, 130, 480, 719)],
}
# Zynq: INT column where the island flow's shell region starts (it spans [SHELL_X0, PS boundary
# + ISLAND_SHELL_STRIP_COLS)); the fabric left of it (above the PS on the xczu7ev) is an extra
# island region. CONTAIN_ROUTING does not hold the nets of the PS8 (even with the PS8 in the
# pblock), and with the 3-column strip of the DynaRapid flow 434 PS<->shell nets ran through
# INT columns 30-34 of the islands; with 5 columns 28 nets remain, nearly all in the next column,
# which is therefore left empty (ISLAND_MOAT_COLS): the islands start right of it
ISLAND_SHELL_STRIP_COLS = int(os.environ.get("FINN_RWI_SHELL_STRIP", "5"))
ISLAND_MOAT_COLS = int(os.environ.get("FINN_RWI_MOAT", "1"))
SHELL_X0 = {"xczu7ev-ffvc1156-2-e": 20, "xczu9eg-ffvb1156-2-e": 17}
# hold margin (ns) of the island routing, see island_tcl
ISLAND_HOLD_MARGIN_NS = float(os.environ.get("FINN_RWI_ISLAND_HOLD_MARGIN", "0.05"))

HERE = os.path.dirname(os.path.abspath(__file__))

# Vivado's link_design of RTL + checkpoints reports a spurious error in 2024.2
# (Designutils 20-50) although the design is linked
_LINK = """if {[catch {link_design -top %s -part %s -mode out_of_context} err]} {
    if {[catch {current_design}] || [llength [get_cells -quiet -hier -filter {IS_BLACKBOX}]] > 0} {error $err}
    puts "INFO: link_design error ignored, design linked: $err"
}"""


def rapidwright_root():
    for d in (
        os.environ.get("RAPIDWRIGHT_PATH"),
        os.path.join(os.environ["FINN_ROOT"], "deps", "RapidWright"),
        os.path.join(os.environ["FINN_ROOT"], "deps", "DynaRapid", "RapidWright"),
    ):
        if d and os.path.isdir(os.path.join(d, "jars")):
            return d
    raise RuntimeError("RapidWright not found (set RAPIDWRIGHT_PATH)")


def stitcher_classpath():
    """Compile IslandStitcher.java (once per source version) against RapidWright."""
    rw = rapidwright_root()
    cp = "%s:%s" % (os.path.join(rw, "bin"), os.path.join(rw, "jars", "*"))
    src = os.path.join(HERE, "IslandStitcher.java")
    out = os.path.join(os.environ["FINN_BUILD_DIR"], "rwislands", "classes")
    cls = os.path.join(out, "IslandStitcher.class")
    if not os.path.isfile(cls) or os.path.getmtime(cls) < os.path.getmtime(src):
        os.makedirs(out, exist_ok=True)
        javac = os.path.join(os.path.dirname(java_bin()), "javac")
        subprocess.run([javac, "-cp", cp, "-d", out, src], check=True)
    return "%s:%s" % (out, cp)


def parse_util(f):
    """Resources of a synthesized component from report_utilization."""
    res = dict.fromkeys(("lut", "lutram", "ff", "carry", "bram", "dsp", "uram"), 0.0)
    if not os.path.isfile(f):
        return res
    txt = open(f, errors="ignore").read()
    pats = {
        "lut": r"\|\s*CLB LUTs\*?\s*\|\s*([\d.]+)",
        "lutram": r"\|\s*LUT as Memory\s*\|\s*([\d.]+)",
        "ff": r"\|\s*CLB Registers\s*\|\s*([\d.]+)",
        "carry": r"\|\s*CARRY8\s*\|\s*([\d.]+)",
        "bram": r"\|\s*Block RAM Tile\s*\|\s*([\d.]+)",
        "dsp": r"\|\s*DSPs\s*\|\s*([\d.]+)",
        "uram": r"\|\s*URAM\s*\|\s*([\d.]+)",
    }
    for k, p in pats.items():
        m = re.search(p, txt)
        if m:
            res[k] = float(m.group(1))
    return res


_CHEAP_OPS = {
    "StreamingFIFO_rtl",
    "StreamingDataWidthConverter_rtl",
    "StreamingDataWidthConverter_hls",
    "FMPadding_rtl",
    "FMPadding_hls",
}


def _synth_estimate(node):
    """Rough synthesis effort of a node (FINN's LUT estimate; unknown -> large)."""
    from qonnx.custom_op.registry import getCustomOp

    try:
        return float(getCustomOp(node).lut_estimation())
    except Exception:
        return 1e9


def synthesize(accel, dcps, work, part, clk_ns, cpus, slots):
    """Out-of-context synthesis of every distinct component, in parallel. Nodes with plain HDL
    are synthesized in shared Vivado sessions (the start and device load are paid once per
    session), the others (weight streamer hierarchies) through a one-node block design.
    Returns {dcp: {"status", "synth_s", "util"}}."""
    synth_dir = os.path.join(work, "synth")
    os.makedirs(synth_dir, exist_ok=True)
    first = {}
    for n in accel.graph.node:
        first.setdefault(dcps[n.name], n)
    # only small, fast node types share sessions; every other node (compute layers, whose
    # synthesis can take minutes, e.g. thresholds in distributed RAM) gets its own Vivado run
    # so that no heavy node waits behind others; longest (estimated) first
    single, cheap = [], []
    for dcp, n in first.items():
        if direct_sources(n) is not None and n.op_type in _CHEAP_OPS:
            cheap.append((dcp, n))
        else:
            single.append((dcp, n))
    res = {}
    # checkpoints left by an interrupted run of the same build directory are reused
    for dcp, n in list(first.items()):
        f = os.path.join(synth_dir, dcp + "_synth.dcp")
        u = os.path.join(synth_dir, dcp + ".util")
        if os.path.isfile(f) and os.path.isfile(u):
            res[dcp] = {"status": "ok", "synth_s": 0.0, "done_s": 0.0, "util": parse_util(u), "reused": True}
    single = [it for it in single if it[0] not in res]
    cheap = [it for it in cheap if it[0] not in res]
    single.sort(key=lambda it: -_synth_estimate(it[1]))
    # sessions of a few cheap nodes each (they queue behind the heavy nodes)
    n_sess = max(1, min(len(cheap), min(cpus, slots)))
    sessions = [cheap[k::n_sess] for k in range(n_sess)] if cheap else []

    def run(tcl_lines, name, items):
        d = os.path.join(work, "components", name)
        os.makedirs(d, exist_ok=True)
        tcl = os.path.join(d, "synth.tcl")
        with open(tcl, "w") as f:
            f.write("\n".join(tcl_lines) + "\n")
        t0 = time.time()
        rc, t = run_vivado(tcl, os.path.join(d, "synth.log"), d)
        for dcp, _ in items:
            ok = os.path.isfile(os.path.join(synth_dir, dcp + "_synth.dcp"))
            res[dcp] = {
                "status": "ok" if ok else "synth_failed",
                "synth_s": t,
                "done_s": time.time() - t0,
                "util": parse_util(os.path.join(synth_dir, dcp + ".util")),
            }

    def single_job(item):
        dcp, n = item
        d = os.path.join(work, "components", dcp)
        # leftovers of an interrupted run (block-design sources are copied in with add_files)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        direct = direct_sources(n)
        if direct is not None:
            body = direct_synth_tcl(
                n, dcp, d, synth_dir, part, 1, direct[0], direct[1], metadata=False, directive=synth_args(part)
            )
        else:
            body = synth_tcl(
                accel, n, dcp, d, synth_dir, part, clk_ns, 1, metadata=False, directive=synth_args(part)
            )
        run(body.splitlines(), dcp, [item])

    def sess_job(items):
        lines = ["set_param general.maxThreads 1"]
        for dcp, n in items:
            d = os.path.join(work, "components", dcp)
            os.makedirs(d, exist_ok=True)
            files, top = direct_sources(n)
            body = direct_synth_tcl(
                n, dcp, d, synth_dir, part, 1, files, top, metadata=False, directive=synth_args(part)
            )
            lines += [l for l in body.splitlines() if not l.startswith("set_param")]
            lines += ["close_design", "remove_files -quiet [get_files -quiet]"]
        run(lines, "session_" + items[0][0], items)

    with ThreadPoolExecutor(max_workers=max(1, min(slots, len(single) + len(sessions)))) as ex:
        futs = [ex.submit(single_job, it) for it in single] + [ex.submit(sess_job, s) for s in sessions]
        for f in futs:
            f.result()
    return res


def island_tcl(name, island_v, dcp_files, part, clk_ns, ranges, cr, threads, out_dir, contain=True, boundary=()):
    """Out-of-context place and route of one island in its pblock (ranges). boundary: ports that
    connect to the shell; they get partition pins on the island's slices, so that their nets are
    routed from the driver / to the load up to the island's edge (an out-of-context port without
    partition pin leaves its net unrouted, and the island's own routing may take all exits of
    its driver: VGG10, 4 IODMA AXI-Lite nets the assembly router could not get out)."""
    steps = profile(part)
    t = [
        "set_param general.maxThreads %d" % threads,
        "set t0 [clock milliseconds]",
        'proc stamp {n} {global t0; puts "STAMP $n [expr ([clock milliseconds] - $t0) / 1000.0]"}',
        "read_verilog %s" % island_v,
    ]
    t += ["read_checkpoint %s" % f for f in dcp_files]
    t += [
        _LINK % (name, part),
        "stamp link",
        "create_clock -period %.3f -name clk [get_ports clk]" % clk_ns,
        # clock source of the out-of-context run (the clock routing is discarded when the
        # island is stitched; Vivado routes the real clock tree at assembly)
        "set bufg [lindex [get_sites -quiet -filter {SITE_TYPE == BUFGCE && CLOCK_REGION == %s}] 0]"
        % cr,
        'if {$bufg == ""} {set bufg [lindex [get_sites -filter {SITE_TYPE == BUFGCE}] 0]}',
        "set_property HD.CLK_SRC $bufg [get_ports clk]",
        "create_pblock pb",
        "resize_pblock pb -add {%s}" % " ".join(ranges),
        "add_cells_to_pblock pb -top",
    ]
    if contain:
        t.append("set_property CONTAIN_ROUTING 1 [get_pblocks pb]")
    if boundary:
        slices = " ".join(r for r in ranges if r.startswith("SLICE"))
        t += [
            "set bports [get_ports -quiet [concat %s]]" % " ".join("{%s} {%s[*]}" % (p, p) for p in boundary),
            "set_property HD.PARTPIN_RANGE {%s} $bports" % slices,
        ]
    t += [
        steps["opt"],
        "stamp opt",
        steps["place"],
        "stamp place",
        steps["phys_opt"],
        "stamp phys_opt",
        # hold margin for the island's own paths: their clock is routed here from a stand-in
        # buffer and re-routed from the shell's clock tree at assembly, which shifts the skew
        # (TFC: 12 island-internal paths at -37..-4 ps after assembly without margin)
        "set_clock_uncertainty -hold %.3f [get_clocks clk]" % ISLAND_HOLD_MARGIN_NS,
        'if {[catch {%s} err]} {puts "INFO: route_design error: $err"}' % steps["route"],
        "stamp route",
    ]
    if steps["post_route_phys_opt"]:
        t += [steps["post_route_phys_opt"], "stamp post_route_phys_opt"]
    t += [
        "report_route_status -file %s/route_status.rpt" % out_dir,
        # the clock routing of the out-of-context run starts at a stand-in BUFGCE; the real
        # clock tree comes from the shell, so leave the clock net to the assembly
        "route_design -unroute -nets [get_nets -of_objects [get_ports clk]]",
        "report_utilization -file %s/utilization.rpt" % out_dir,
        "write_checkpoint -force %s/%s_routed.dcp" % (out_dir, name),
        "write_edif -force %s/%s_routed.edf" % (out_dir, name),
        "stamp write",
    ]
    return "\n".join(t) + "\n"


def top_tcl(top_v, part, out_dir):
    return "\n".join(
        [
            "set_param general.maxThreads 2",
            "read_verilog %s" % top_v,
            "synth_design -top %s -part %s -mode out_of_context" % (TOP_MODULE, part),
            "write_checkpoint -force %s/top_bb.dcp" % out_dir,
            "write_edif -force %s/top_bb.edf" % out_dir,
        ]
    ) + "\n"


def choose_islands(n_nodes, total_cost, slots, islands):
    if islands not in (None, "auto"):
        return int(islands)
    # every Vivado run costs ~1 min of fixed effort on the xczu7ev: islands of at least
    # ~6000 cost units (LUT + FF/2 + ...), no more islands than parallel Vivado runs
    return max(1, min(n_nodes, slots, int(total_cost // 6000) + 1))


def merge_small(segs, costs, min_frac=0.15):
    """Merge islands (contiguous segments (a, b) of the chain) smaller than min_frac of the
    largest one into a neighbour, as long as the merged island stays within the largest. A
    dominant node fixes the largest island; the min-max partition is indifferent to how the rest
    is cut and, for exactly k islands, leaves tiny ones (VGG10, 20 islands: 16 of cost 66-900,
    packed into single columns, whose wide buses the stitcher could not reach)."""
    segs = list(segs)
    cost = lambda s: sum(costs[s[0] : s[1]])
    while len(segs) > 1:
        top = max(cost(s) for s in segs)
        cand = []
        for i, s in enumerate(segs):
            if cost(s) >= min_frac * top:
                continue
            for j in (i - 1, i + 1):
                if 0 <= j < len(segs) and cost(s) + cost(segs[j]) <= top:
                    cand.append((cost(s) + cost(segs[j]), i, j))
        if not cand:
            break
        _, i, j = min(cand)
        a, b = min(i, j), max(i, j)
        segs[a : b + 1] = [(segs[a][0], segs[b][1])]
    return segs


def plan_islands(names, node_res, dev, region, first, slots, islands, res, packing="snake", anchor=None, boundary=None):
    """Cut the node chain (names, in topological order; node_res: synthesized resources per
    node) into islands and floorplan them in the region (list of tile rectangles) plus, for the
    snake, the extra rectangles first. Tries island counts and utilizations (see
    islands_and_stitch); falls back to one island per region rectangle, then to a single island.
    Fills res["islands"], res["floorplan"], res["floorplan_tries"]; returns ({island: member
    names}, pblock ranges per island). anchor: the shell's (x, y) tile position; islands with
    a node in boundary (nodes with ports to the shell, e.g. the IODMAs) are pulled towards it
    (skyline packing)."""
    costs = [island_cost(node_res[n]) for n in names]
    rect_list = region if isinstance(region, list) else [region]

    def anchors_of(isl_k):
        if anchor is None or not boundary:
            return None
        return [anchor if set(mem) & boundary else None for mem in isl_k.values()]

    def islands_of(k, exact=False):
        segs = merge_small(partition(costs, k, exact), costs)
        isl_k = {"island_%d" % i: names[a:b] for i, (a, b) in enumerate(segs)}
        res_k = []
        for mem in isl_k.values():
            tot = {}
            for n in mem:
                for kk, v in node_res[n].items():
                    tot[kk] = tot.get(kk, 0) + v
            res_k.append(tot)
        return isl_k, res_k

    # island counts to try: the automatic (or requested) one, then halving; the snake first
    # (fast) at <= 0.8 utilization for every count, then the 2D packing for small counts
    k0 = choose_islands(len(names), sum(costs), slots, islands)
    ks = []
    k = k0
    while k >= 1:
        ks.append(k)
        k //= 2
    ks = sorted(set(ks), reverse=True)
    plan, tried = None, []
    if packing == "skyline":
        # lowest utilization first, at each level the fewest islands (exact counts: more,
        # smaller islands pack better); scarcest-first order for the search, chain order (closer
        # neighbours) if it fits at the same level
        kmax = min(len(names), max(k0, slots))
        kc = sorted({min(kmax, max(1, int(round(k0 * f)))) for f in (1, 1.5, 2, 3)} | ({20, 28, 40} if kmax >= 20 else set()))
        kc = [k for k in kc if k <= kmax]

        def fewest(u, reg):
            # per count: first the fewest islands reaching the minimal largest island (a dominant
            # node bounds the largest island anyway; more islands would only add tiny ones, each
            # a Vivado run and stitching work), then exactly k (smaller islands pack better)
            for k in kc:
                for exact in (False, True):
                    isl_k, res_k = islands_of(k, exact=exact)
                    try:
                        floorplan(dev, res_k, reg, allocators=("skyline_hard",), utils=(u,), anchors=anchors_of(isl_k))
                        return k, exact
                    except RuntimeError:
                        tried.append((u, k, exact, "skyline", len(reg)))
            return None

        # the main region first; then with the extra rectangles (e.g. the fabric above the PS,
        # whose nets to the main region cross the shell: the stitcher leaves them to the
        # assembly's router), for designs that need its resources (CNV-w1a1 PE=SIMD=1: 222 BRAM36)
        levels = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
        for reg in [region] + ([region + list(first)] if first else []):
            for li, u in enumerate(levels):
                found = fewest(u, reg)
                if found is None:
                    continue
                k, exact = found
                # the next level if it needs at most half the islands (the stitching cost grows
                # with the island count: 82 islands took 555 s to stitch)
                if li + 1 < len(levels):
                    f2 = fewest(levels[li + 1], reg)
                    if f2 is not None and len(islands_of(*f2)[0]) * 2 <= len(islands_of(k, exact)[0]):
                        u, (k, exact) = levels[li + 1], f2
                isl, isl_res = islands_of(k, exact=exact)
                # chain order if it fits at this level (else floorplan falls back to scarcest first)
                plan = floorplan(dev, isl_res, reg, allocators=("skyline",), utils=(u,), anchors=anchors_of(isl))
                break
            if plan is not None:
                break
        res["skyline"] = {"candidates": kc}
    for allocs, kmax, utils in ((("snake",), None, (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)), (("rects",), 8, None)):
        if plan is not None:
            break
        for k in ks:
            if kmax is not None and k > kmax:
                continue
            isl, isl_res = islands_of(k)
            try:
                plan = floorplan(dev, isl_res, region, first_lanes=first, allocators=allocs, utils=utils)
                break
            except RuntimeError:
                tried.append((k, len(isl), allocs[0]))
        if plan is not None:
            break
    res["floorplan_tries"] = tried
    if plan is None and len(rect_list) > 1:
        # one island per region rectangle (e.g. per SLR), chain cut by cost
        from finn.util.rwislands.device import capacity, pblock_ranges
        from finn.util.rwislands.floorplan import need

        isl, isl_res = islands_of(len(rect_list))
        util = {"lut": 0.9, "bram": 1.0, "dsp": 1.0, "uram": 1.0}
        if len(isl) == len(rect_list) and all(
            all(capacity(dev.sites_in(*rc))[kk] >= v for kk, v in need(r, util).items())
            for r, rc in zip(isl_res, rect_list)
        ):
            res["floorplan_fallback"] = "one island per region rectangle"
            lanes = list(rect_list)
            plan = ([[rc] for rc in rect_list], [pblock_ranges(dev.sites_in(*rc)) for rc in rect_list], util, lanes)
    try:
        if plan is None:
            raise RuntimeError("the islands do not fit into the region %s (tried %s)" % (region, tried))
        rects, ranges, util, lanes = plan
    except RuntimeError as e:
        # too dense for one rectangle per island (e.g. BRAM/URAM in few columns): the whole
        # accelerator as one island in the main region (one connected rectangle, so its
        # routing stays contained; without containment its routing used shell tiles and the
        # assembly lost the shell's placement)
        from finn.util.rwislands.device import pblock_ranges

        res["floorplan_fallback"] = str(e)
        isl = {"island_0": names}
        tot = {}
        for n in names:
            for kk, v in node_res[n].items():
                tot[kk] = tot.get(kk, 0) + v
        isl_res = [tot]
        # the first rectangle of the region (one SLR on Alveo)
        lanes = [region[0] if isinstance(region, list) else region]
        rects = [lanes]
        ranges = [[g for r in lanes for g in pblock_ranges(dev.sites_in(*r))]]
        util = None
    res["islands"] = {
        name: {"nodes": mem, "res": r, "cost": sum(costs[names.index(n)] for n in mem), "rects": rc}
        for (name, mem), r, rc in zip(isl.items(), isl_res, rects)
    }
    res["floorplan"] = {"util": util, "lanes": lanes, "region": region}
    return isl, ranges


def islands_and_stitch(
    accel, g, dcps, synth, dev, part, clk_ns, work, cpus, slots, islands, region, first, res, stamp,
    # the stitch only pre-routes the inter-island nets; pins boxed in by island routing (which
    # RWRoute keeps) stay unrouted after a few iterations and are routed at assembly, where
    # Vivado may rip up island routes (MobileNet: stuck at 10 overlaps from iteration 11 to 29,
    # ~78 s per iteration)
    rwroute_max_iter=10,
    max_island_overlaps=0,
    packing="snake",
    stitch_region=None,
    anchor=None,
):
    """Islands, floorplan (region + first lanes), island P&R in parallel with the top synthesis,
    RapidWright stitching. Fills res; returns (failure status or None, stitched accelerator dcp).
    packing "snake": consecutive islands in lanes; "skyline": variable-size staircase islands
    packed by resource mix (floorplan.allocate_skyline), searched over the utilization and the
    island count. stitch_region: site ranges the inter-island routes must stay in (the island
    region; the stitched design does not know the shell's routing)."""
    names = [n.name for n in accel.graph.node]
    node_res = {n: synth[dcps[n]]["util"] for n in names}
    # 2. islands and floorplan
    # nodes with ports to the shell (IODMAs, compute nodes with AXI-Lite): their islands go next
    # to the shell
    boundary = {n for n in names if len(boundary_ports(g, [n])) > 1}
    isl, ranges = plan_islands(
        names, node_res, dev, region, first, slots, islands, res, packing, anchor=anchor, boundary=boundary
    )
    stamp("floorplan")

    # 3. island place and route + accelerator top, in parallel
    isl_dir = os.path.join(work, "islands")
    threads = max(1, min(8, cpus // max(1, len(isl))))
    synth_dir = os.path.join(work, "synth")

    def island_job(name):
        mem = isl[name]
        d = os.path.join(isl_dir, name)
        os.makedirs(d, exist_ok=True)
        v = os.path.join(d, name + ".v")
        with open(v, "w") as f:
            f.write(island_verilog(name, g, mem, dcps))
        files = [os.path.join(synth_dir, dc + "_synth.dcp") for dc in sorted({dcps[n] for n in mem})]
        r0 = res["islands"][name]["rects"][0]
        cr = next(s.cr for s in dev.sites_in(*r0) if s.type.startswith("SLICE"))
        tcl = os.path.join(d, "island.tcl")
        with open(tcl, "w") as f:
            f.write(
                island_tcl(
                    name, v, files, part, clk_ns, ranges[list(isl).index(name)], cr, threads, d,
                    boundary=boundary_ports(g, mem),
                )
            )
        rc, t = run_vivado(tcl, os.path.join(d, "island.log"), d)
        info = {"pnr_s": t, "rc": rc}
        info.update(_island_reports(d))
        info["stamps"] = {
            kk: float(vv)
            for kk, vv in re.findall(r"^STAMP (\w+) ([\d.]+)", open(os.path.join(d, "island.log")).read(), re.M)
        }
        info["dcp"] = os.path.join(d, name + "_routed.dcp")
        # a few nets with overlaps are acceptable when the final route_design re-routes the
        # whole design with only the islands' placement locked (Alveo flows)
        errs = info.get("routing_errors")
        info["ok"] = rc == 0 and os.path.isfile(info["dcp"]) and errs is not None and errs <= max_island_overlaps
        return name, info

    def top_job():
        d = os.path.join(work, "top")
        os.makedirs(d, exist_ok=True)
        v = os.path.join(d, "top.v")
        with open(v, "w") as f:
            f.write(top_verilog(g, isl))
        tcl = os.path.join(d, "top.tcl")
        with open(tcl, "w") as f:
            f.write(top_tcl(v, part, d))
        rc, t = run_vivado(tcl, os.path.join(d, "top.log"), d)
        return rc, t, d

    with ThreadPoolExecutor(max_workers=max(1, min(slots, len(isl) + 1))) as ex:
        f_top = ex.submit(top_job)
        # largest islands first
        order = sorted(isl, key=lambda n: -res["islands"][n]["cost"])
        infos = dict(f.result() for f in [ex.submit(island_job, n) for n in order])
        top_rc, top_s, top_dir = f_top.result()
    stamp("islands")
    for name, info in infos.items():
        res["islands"][name].update(info)
    res["top_s"] = top_s
    bad = [n for n, i in infos.items() if not i["ok"]]
    if bad or top_rc != 0:
        res.update(failed=bad, top_rc=top_rc)
        return "island_failed", None

    # 4. stitch with RapidWright
    st_dir = os.path.join(work, "stitch")
    os.makedirs(st_dir, exist_ok=True)
    accel_dcp = os.path.join(st_dir, "accel_routed.dcp")
    cmd = [
        java_bin(),
        "-Xmx32G",
        "-cp",
        stitcher_classpath(),
        "IslandStitcher",
        os.path.join(top_dir, "top_bb.dcp"),
        os.path.join(top_dir, "top_bb.edf"),
        accel_dcp,
        str(rwroute_max_iter),
        str(max(1, min(16, cpus))),
    ]
    if stitch_region:
        cmd.append("--region=" + stitch_region)
    cmd += ["%s=%s,%s" % (n, i["dcp"], i["dcp"].replace(".dcp", ".edf")) for n, i in infos.items()]
    t0 = time.time()
    log = os.path.join(st_dir, "stitch.log")
    env = dict(os.environ, RW_QUIET_MESSAGE="1")
    with open(log, "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    res["stitch_s"] = time.time() - t0
    txt = open(log, errors="ignore").read()
    res["stitch_stamps"] = {kk: float(vv) for kk, vv in re.findall(r"^STAMP (\w+) ([\d.]+)", txt, re.M)}
    m = re.search(r"RESULT unrouted_pins (\d+)", txt)
    res["stitch_unrouted_pins"] = int(m.group(1)) if m else None
    res["stitch_unrouted_nets"] = re.findall(r"^UNROUTED_NET (\S+)", txt, re.M)
    # PIPs used by two nets after stitching (RapidWright's check): islands whose contained
    # routing overlaps (shared INT columns) or stitch routes over island routing
    res["stitch_pip_conflicts"] = len(re.findall(r"^pip \S+ users = ", txt, re.M))
    stamp("stitch")
    if rc != 0 or not os.path.isfile(accel_dcp):
        return "stitch_failed", None

    return None, accel_dcp


def rw_islands_zynq_build(
    kernel_models,
    board,
    part,
    clk_ns,
    out_dir,
    shell_lib,
    islands="auto",
    workers=None,
    rwroute_max_iter=10,
):
    """Build the bitfile of a ZynqBuild design (partition models with generated IP).
    Returns a result dict with the bitfile, hwh and per-stage times."""
    os.makedirs(out_dir, exist_ok=True)
    cpus = workers or usable_cpus()
    _, slots = vivado_slots()
    work = os.path.join(out_dir, "work")
    res = {"out_dir": out_dir, "board": board, "part": part, "clk_ns": clk_ns, "flow": "rwislands"}
    stamps = {}
    t_total = time.time()

    def stamp(k):
        stamps[k] = time.time() - t_total

    accel = merge_partitions(kernel_models)
    accel.save(os.path.join(out_dir, "accel.onnx"))
    vendor = {n.name: vendor_ip_cores(n) for n in accel.graph.node}
    vendor = {k: v for k, v in vendor.items() if v}
    if vendor:
        res.update(status="unsupported_vendor_ip", vendor_ip=vendor)
        return _done(res, stamps, out_dir, t_total)
    ports = mm_ports(accel)
    assert ports, "the island shell flow needs IODMAs at the accelerator boundary"
    g = channel_graph(accel)
    dcps = {n.name: component_name(accel, n, part, clk_ns) for n in accel.graph.node}
    dev = load_device(part)

    # 1. shell (cached) and node synthesis, concurrently
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_shell = ex.submit(
            build_shell,
            board,
            part,
            clk_ns,
            ports,
            shell_lib,
            max(1, min(8, cpus // 4)),
            SHELL_X0.get(part, 0),
            ISLAND_SHELL_STRIP_COLS,
        )
        synth = synthesize(accel, dcps, work, part, clk_ns, cpus, slots)
        stamp("synth")
        shell_dir, shell_res = f_shell.result()
    stamp("shell")
    res["shell"] = shell_res
    res["synth"] = synth
    failed = [d for d, r in synth.items() if r["status"] != "ok"]
    if failed:
        res.update(status="synth_failed", failed=failed)
        return _done(res, stamps, out_dir, t_total)
    if shell_res["status"] not in ("built", "cached"):
        res["status"] = "shell_failed"
        return _done(res, stamps, out_dir, t_total)

    # the assembly Vivado opens the shell now and waits for the stitched accelerator
    asm_dir = os.path.join(out_dir, "assembly")
    os.makedirs(asm_dir, exist_ok=True)
    bitfile = os.path.join(out_dir, "resizer.bit")
    accel_dcp = os.path.join(work, "stitch", "accel_routed.dcp")
    trigger = os.path.join(asm_dir, "accel_ready")
    if os.path.exists(trigger):
        os.remove(trigger)
    asm_tcl = os.path.join(asm_dir, "assemble.tcl")
    # the full route_design by default: once the shell and the islands do not overlap, it only
    # routes the boundary/clock/static nets, at a fixed cost (VGG10: RT build + timing init ~80 s,
    # hold fixing ~70 s); the interactive router (incremental) saves little on top and fails on
    # single nets it cannot finish (VGG10: after 170 s)
    incremental = os.environ.get("FINN_RWI_ASM_ROUTE", "full") == "incremental"
    with open(asm_tcl, "w") as f:
        f.write(
            assemble_tcl(
                shell_dir, accel_dcp, asm_dir, bitfile, part, threads=min(16, cpus), trigger=trigger,
                incremental=incremental,
            )
        )
    asm_pool = ThreadPoolExecutor(max_workers=1)
    f_asm = asm_pool.submit(run_vivado, asm_tcl, os.path.join(asm_dir, "assemble.log"), asm_dir)

    def abort(status, **kw):
        with open(trigger, "w") as f:
            f.write("abort")
        f_asm.result()
        res.update(status=status, **kw)
        return _done(res, stamps, out_dir, t_total)

    # 2.-4. islands, floorplan, island P&R, stitching
    region, first = island_region(part, dev)
    # skyline packing (compact variable-size islands) by default; the snake (consecutive islands
    # in ~10-column lanes) made few, dense, partly L-shaped islands on the xczu7ev (VGG10: 4
    # islands at 75 % LUTs, one router stuck on a single overlap for ~250 s)
    # (the skyline packs the main region only: the fabric above the PS is cut off from it by the
    # shell, whose routing the stitcher does not know; the snake fallback still uses it)
    packing = os.environ.get("FINN_RWI_PACKING", "skyline")
    rects = region + first
    stitch_region = " ".join(rg for r in rects for rg in pblock_ranges(dev.sites_in(*r)))
    fail, accel_dcp2 = islands_and_stitch(
        accel, g, dcps, synth, dev, part, clk_ns, work, cpus, slots, islands, region, first, res,
        stamp, rwroute_max_iter, packing=packing, stitch_region=stitch_region, anchor=shell_anchor(part, dev),
    )
    if fail is not None:
        return abort(fail)
    assert accel_dcp2 == accel_dcp

    # 5. assembly (the shell is already open in the waiting Vivado)
    t0 = time.time()
    with open(trigger, "w") as f:
        f.write("go full" if res.get("stitch_unrouted_pins") else "go")
    rc, _ = f_asm.result()
    asm_pool.shutdown()
    res["assembly_s"] = time.time() - t0
    log = os.path.join(asm_dir, "assemble.log")
    stamp("assembly")
    res["assembly_stamps"] = {
        kk: float(vv) for kk, vv in re.findall(r"^STAMP (\w+) ([\d.]+)", open(log).read(), re.M)
    }
    res.update(_reports(asm_dir))
    res["timing_rpt"] = os.path.join(asm_dir, "timing_summary.rpt")
    res["utilization_xml"] = os.path.join(asm_dir, "synth_report.xml")
    if rc != 0 or not os.path.isfile(bitfile):
        res["status"] = "assembly_failed"
        return _done(res, stamps, out_dir, t_total)
    hwh = os.path.join(out_dir, "resizer.hwh")
    if os.path.isfile(os.path.join(shell_dir, "top.hwh")):
        shutil.copy(os.path.join(shell_dir, "top.hwh"), hwh)
    else:
        hwh = None  # Vivado-only Alveo shell: no PYNQ hardware handoff
    res.update(status="ok", bitfile=bitfile, hwh=hwh)
    return _done(res, stamps, out_dir, t_total)


def shell_anchor(part, dev):
    """(x, y) tile position of the shell's interface to the islands, or None: the Alveo shell
    sits in the clock regions right of SLR0's island rectangle (around the PCIe block); islands
    in other SLRs are two crossings away. On the Zynq parts every island is next to the shell
    strip anyway, and pulling the IODMA islands there made VGG10's packing worse (6 islands at
    80 % LUTs instead of 4 at 70 %)."""
    if part in VIVADO_ALVEO_REGION:
        x0, x1, y0, y1 = VIVADO_ALVEO_REGION[part][0]
        return (x1, (y0 + y1) // 2)
    return None


def island_region(part, dev):
    """Island region of a part next to the island flow's shell: (main rectangles, extra
    rectangles), each (x0, x1, y0, y1) in tile coordinates. Alveo (Vivado-only shell around the
    PCIe block): one rectangle per SLR. Zynq: right of the shell strip, plus the fabric left of the
    shell above the PS."""
    if part in VIVADO_ALVEO_REGION:
        return list(VIVADO_ALVEO_REGION[part]), []
    x0 = PS_BOUNDARY_INT_X[part] + ISLAND_SHELL_STRIP_COLS + ISLAND_MOAT_COLS
    region = [(x0, dev.xmax, 0, dev.ymax)]
    sx0 = SHELL_X0.get(part, 0)
    first = []
    if sx0 > 0:
        # its rows are those with sites there
        ys = [st.y for st in dev.sites if st.x < sx0]
        first = [(0, sx0 - 1, min(ys) - min(ys) % 5, dev.ymax)]
    return region, first


def _island_reports(d):
    """Routing errors of an island: conflicts and unrouted nets. Nets ending in a partition pin
    (boundary ports, routed up to the island's edge) count as antennas and those with a port
    pin as unplaced: both are expected out of context."""
    res = {}
    rs = os.path.join(d, "route_status.rpt")
    if os.path.isfile(rs):
        txt = open(rs).read()

        def count(what):
            m = re.search(r"# of nets with %s\.+ :\s+(\d+)" % what, txt)
            return int(m.group(1)) if m else 0

        if re.search(r"# of nets with routing errors", txt):
            res["routing_errors"] = count("routing errors") - count("antennas/islands") - count("some unplaced pins")
            res["boundary_stubs"] = count("antennas/islands")
        else:
            res["routing_errors"] = None
    return res


def _done(res, stamps, out_dir, t_total):
    res["stamps"] = stamps
    res["total_s"] = time.time() - t_total
    with open(os.path.join(out_dir, "rwislands_zynq.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res
