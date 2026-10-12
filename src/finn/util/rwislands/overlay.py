# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tile overlay: the accelerator region of a cached shell cut into a snake of clock-region tiles,
each a DFX reconfigurable partition, with fixed link bundles between snake neighbours through
static anchor registers (links.py). Built once per (shell, grid, link widths) and cached; per model
every used tile is implemented on its own, in its abstract shell, in parallel, down to a partial
bitstream - no step processes the whole design (RapidStream 2.0 / PRflow / PLD, RELATED_WORK.md).

Tile k's module (tile<k>) ports: clk, rst; west link (towards tile k-1) w_in[WF] (forward stream
bits into the tile), w_out[WB] (backward bits out); east link (towards tile k+1) e_out[WF], e_in[WB];
tile 0 also has the shell boundary ports of the accelerator (the bridges' register slices are the
static timing points there, so no anchors). Forward = along the snake (away from tile 0).
"""

import hashlib
import json
import os
import re
import warnings

from finn.util.dynarapid.shell import CORE_MODULE, _bus
from finn.util.dynarapid.tools import run_vivado, vivado_version
from finn.util.rwislands.device import pblock_ranges
from finn.util.rwislands.floorplan import grid_tiles
from finn.util.rwislands.links import ANCHOR

OVERLAY_VERSION = 6
# DSP / BRAM / URAM share of a tile's sites a model may use (knob hard=<percent>). Was 0.6: with
# tiles abutting on clock-region rows, routing containment blocked hard blocks at their edges (VGG10
# 0.25x, 1-region tile 5 at the device edge: 19 of 54 DSP sites blocked). With the static gaps around
# the tiles, a 2x2-region tile (22-tile overlay, tile 7) placed and routed 352 connected DSPs = all
# sites of its snapped pblock (96 % of the nominal 368; BRAM36 66 of 69): VGG10 8x needs 65 % of
# the overlay's DSPs.
HARD_UTIL = 0.9
# write_bitstream fails with more reconfigurable partitions in the design (Designutils 20-1700
# "bitset::set: __position (which is 32) >= _Nb (which is 32)", also for a single tile's partial
# bitstream, Vivado 2024.2 and 2025.1; the 79-tile static placed and routed fine): the snake is
# cut after MAX_TILES tiles
MAX_TILES = 32


def overlay_knobs():
    """FINN_RWI_OVERLAY: "cols=16,rows=60,wf=1024,wb=256,boundary_rows=120" - tile grid (as
    FINN_RWI_GRID, clock-region aligned), link widths forward / backward, the boundary tile's
    height (tile 0 holds the IODMAs and all shell boundary signals: one clock region did not route
    1772 of them, scratchpad DFX test 2026-10-10), tiles: only the first N snake tiles (0: all),
    gap_cols / gap_rows: tile columns / rows left to the static design on each side of a tile,
    crs: clock-region columns per tile; per model (not part of the overlay's key): hard: DSP / BRAM /
    URAM share of a tile a model may use (percent), part: size of a split layer's part (percent of a
    tile), pipe: link pipelining depth (links.link_regs; registers in transit tiles from 2 on),
    tile0: compute nodes may share tile 0 with the IODMAs, plan: dp (plan_overlay_dp, default) | greedy
    (plan_overlay, the planner of the first results), umin: the dp planner's lowest target logic
    utilization (percent)."""
    kn = {"cols": 16, "rows": 60, "order": "rows", "wf": 1024, "wb": 256, "boundary_rows": 120, "tiles": 0,
          "gap_cols": 1, "gap_rows": 1, "crs": 1, "hard": int(HARD_UTIL * 100), "part": 45, "pipe": 1, "tile0": 1, "plan": "dp", "umin": 35}
    for kv in filter(None, os.environ.get("FINN_RWI_OVERLAY", "").split(",")):
        k, v = kv.split("=")
        kn[k] = v if k in ("order", "plan") else int(v)
    return kn


def overlay_tiles(dev, region, kn, anchor):
    """Tile rectangles in snake order; tile 0 (next to the shell) is boundary_rows high."""
    tiles = grid_tiles(dev, region, cols=kn["cols"], rows=kn["rows"], order=kn["order"],
                       start="right" if anchor is not None else "origin", align="cr", crs=kn.get("crs", 1))
    if kn["boundary_rows"] > kn["rows"] and tiles:
        x0, x1, y0, y1 = tiles[0]
        y1b = y0 + kn["boundary_rows"] - 1
        # tile 0 grows upwards; the tiles it covers leave the snake
        tiles = [(x0, x1, y0, y1b)] + [t for t in tiles[1:] if not (t[0] <= x1 and t[1] >= x0 and t[2] <= y1b and t[3] >= y0)]
    # tiles=N: only the first N tiles of the snake (a smaller overlay for smaller models)
    tiles = tiles[:kn["tiles"]] if kn.get("tiles") else tiles
    if len(tiles) > MAX_TILES:
        warnings.warn("overlay: %d tiles, cut to the first %d of the snake (bitstream limit)" % (len(tiles), MAX_TILES))
        tiles = tiles[:MAX_TILES]
    # every tile pblock shrunk by gap_cols / gap_rows on each side: the gaps are static channels
    # for the link anchors (abutting tiles left them no static sites nearby; RapidStream reserves
    # anchor columns between its islands for the same reason), and the tile edges no longer lie
    # on clock-region boundaries (Constraints 18-5648: limited routability at such edges). (The
    # place_design "hang" from ~16 tiles on was the grey boxes' detail placement: PLACE_GREYBOX_TCL)
    gc, gr = kn.get("gap_cols", 0), kn.get("gap_rows", 0)
    return [(x0 + gc, x1 - gc, y0 + gr, y1 - gr) for x0, x1, y0, y1 in tiles]


def overlay_key(shell_dir, part, tiles, kn):
    sig = {"version": OVERLAY_VERSION, "vivado": vivado_version(), "shell": os.path.basename(shell_dir),
           "part": part, "tiles": tiles, "wf": kn["wf"], "wb": kn["wb"]}
    return "overlay" + hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]


def boundary_ports(ports):
    """(declarations, inputs [(name, width)], outputs [(name, width)]) of the shell boundary."""
    dec, ins, outs = [], [], []
    for d in ports:
        for c in d["in"]:
            dec += ["input %s%s" % (_bus(c["width"]), c["data"]), "input %s" % c["valid"], "output %s" % c["ready"]]
            ins += [(c["data"], c["width"]), (c["valid"], 1)]
            outs += [(c["ready"], 1)]
        for c in d["out"]:
            dec += ["output %s%s" % (_bus(c["width"]), c["data"]), "output %s" % c["valid"], "input %s" % c["ready"]]
            outs += [(c["data"], c["width"]), (c["valid"], 1)]
            ins += [(c["ready"], 1)]
    return dec, ins, outs


def tile_ports(k, n, kn, ports):
    """Port declarations of tile k of n."""
    p = ["input clk", "input rst"]
    if k > 0:
        p += ["input [%d:0] w_in" % (kn["wf"] - 1), "output [%d:0] w_out" % (kn["wb"] - 1)]
    if k < n - 1:
        p += ["output [%d:0] e_out" % (kn["wf"] - 1), "input [%d:0] e_in" % (kn["wb"] - 1)]
    if k == 0:
        p += boundary_ports(ports)[0]
    return p


def container_verilog(ports, n, kn):
    """finn_accel_core (the shell's accelerator cell): the tiles as black boxes, anchor registers
    on every link wire between snake neighbours, the boundary straight to tile 0."""
    dec, ins, outs = boundary_ports(ports)
    t = ["module %s (" % CORE_MODULE, ",\n".join("    " + x for x in ["input clk", "input rst"] + dec), ");"]
    for k in range(n - 1):
        t += ["    wire [%d:0] f%d_a, f%d_b;" % (kn["wf"] - 1, k, k), "    wire [%d:0] b%d_a, b%d_b;" % (kn["wb"] - 1, k, k),
              "    rwi_anchor #(.W(%d)) af%d (.clk(clk), .d(f%d_a), .q(f%d_b));" % (kn["wf"], k, k, k),
              "    rwi_anchor #(.W(%d)) ab%d (.clk(clk), .d(b%d_a), .q(b%d_b));" % (kn["wb"], k, k, k)]
    # the reset runs along the snake through one static register per edge (tile k sees it k
    # cycles late): a direct wire from the shell's reset to the far tiles took 14.7 ns (VGG10 8x,
    # 22 tiles, tiles 14-21 failed setup on their first reset stage). The tiles only talk through
    # the link handshakes, so they need not leave reset in the same cycle
    # (a chain register driving both the next one and a tile failed DFX routing, Constraints
    # 18-4430: every tile gets its own register off the chain, so that each boundary net has one
    # static driver and one partition pin, like the link anchors)
    # (DONT_TOUCH: synthesis merged the equivalent chain and tile registers back into one)
    t += ["    wire r0 = rst;"] + ["    wire r%d; (* DONT_TOUCH = \"yes\" *) rwi_anchor #(.W(1)) ar%d (.clk(clk), .d(r%d), .q(r%d));"
                                  % (k + 1, k, k, k + 1) for k in range(n - 1)]
    t += ["    wire rt%d; (* DONT_TOUCH = \"yes\" *) rwi_anchor #(.W(1)) at%d (.clk(clk), .d(r%d), .q(rt%d));" % (k, k, k, k)
          for k in range(n)]
    for k in range(n):
        c = ["    .clk(clk)", "    .rst(rt%d)" % k]
        if k > 0:
            c += ["    .w_in(f%d_b)" % (k - 1), "    .w_out(b%d_a)" % (k - 1)]
        if k < n - 1:
            c += ["    .e_out(f%d_a)" % k, "    .e_in(b%d_b)" % k]
        if k == 0:
            c += ["    .%s(%s)" % (nm, nm) for nm, _ in ins + outs]
        t.append("    (* keep_hierarchy = \"yes\" *) tile%d t%d (\n%s\n    );" % (k, k, ",\n".join(c)))
    t.append("endmodule")
    stubs = [grey_box("tile%d" % k, tile_ports(k, n, kn, ports)) for k in range(n)]
    return "\n".join(t) + "\n" + "\n".join(stubs) + "\n" + ANCHOR


def grey_box(name, decls):
    """A tile's empty configuration, written out (what update_design -buffer_ports makes of a
    black box; in the full shell design that took ~97 s per tile, and one call for several tiles
    fails on their different interfaces, Netlist 29-77): a LUT1 driving each output low, a LUT1
    load on each input, a register on the clock, all DONT_TOUCH."""
    body = []
    for d, msb, p in re.findall(r"(input|output)\s+(?:\[(\d+):0\]\s*)?(\w+)", "\n".join(decls)):
        w = int(msb) + 1 if msb else 1
        if p == "clk":
            body.append("    (* DONT_TOUCH = \"TRUE\" *) FDCE gb_clk (.C(clk), .CE(1'b0), .CLR(1'b0), .D(1'b0), .Q());")
            continue
        body.append("    for (i = 0; i < %d; i = i + 1) begin : gb_%s" % (w, p))
        if d == "output":
            body.append("        (* DONT_TOUCH = \"TRUE\" *) LUT1 #(.INIT(2'b00)) l (.I0(1'b0), .O(%s%s));" % (p, "[i]" if msb else ""))
        else:
            body.append("        (* DONT_TOUCH = \"TRUE\" *) LUT1 #(.INIT(2'b10)) l (.I0(%s%s), .O());" % (p, "[i]" if msb else ""))
        body.append("    end")
    return "module %s (\n%s\n);\n    genvar i;\n%s\nendmodule" % (name, ",\n".join("    " + x for x in decls), "\n".join(body))


# the grey boxes placed by hand before place_design: left to the placer, its detail placement of
# the ~2.4k unconnected DONT_TOUCH LUT1s per tile grew super-linearly with the tile count (10
# tiles: 4 min in phase 3.3, 24 tiles: 29 min, 22 large tiles: > 30 min in Area Swap - the
# "hang"); pre-placed, place_design of the 22-tile static took 5.5 min. Each link's LUTs go to
# the slices nearest to the neighbour tile (they pull the partition pins to that edge).
PLACE_GREYBOX_TCL = r"""
proc site_xy {s} { regexp {X(\d+)Y(\d+)} $s - x y; list $x $y }
proc box_of {pb} {
    set xs {}; set ys {}
    foreach s [get_sites -of $pb -filter {SITE_TYPE =~ SLICE*}] { lassign [site_xy $s] x y; lappend xs $x; lappend ys $y }
    set xs [lsort -integer $xs]; set ys [lsort -integer $ys]
    list [lindex $xs 0] [lindex $xs end] [lindex $ys 0] [lindex $ys end]
}
proc place_greybox {rp n} {
    set t0 [clock milliseconds]
    set total 0
    for {set k 0} {$k < $n} {incr k} {
        set c [get_cells $rp/t$k]
        set sites [get_sites -of [get_pblocks pb_t$k] -filter {SITE_TYPE =~ SLICE* && IS_USED == 0}]
        set free [dict create]
        foreach s $sites { dict set free $s 1 }
        set pairs {}
        foreach {side nb} [list w [expr {$k - 1}] e [expr {$k + 1}] x -1] {
            if {$side eq "x"} {
                set luts [get_cells -quiet -hier -filter "NAME =~ $c/gb_* && NAME !~ $c/gb_w_* && NAME !~ $c/gb_e_* && REF_NAME == LUT1"]
                set order [dict keys $free]
            } else {
                if {$nb < 0 || $nb >= $n} continue
                set luts [get_cells -quiet -hier -filter "NAME =~ $c/gb_${side}_* && REF_NAME == LUT1"]
                if {![llength $luts]} continue
                lassign [box_of [get_pblocks pb_t$nb]] bx0 bx1 by0 by1
                set dist {}
                foreach s [dict keys $free] {
                    lassign [site_xy $s] x y
                    set dx [expr {$x < $bx0 ? $bx0 - $x : ($x > $bx1 ? $x - $bx1 : 0)}]
                    set dy [expr {$y < $by0 ? $by0 - $y : ($y > $by1 ? $y - $by1 : 0)}]
                    lappend dist [list $s [expr {$dx + $dy}]]
                }
                set order {}
                foreach e [lsort -integer -index 1 $dist] { lappend order [lindex $e 0] }
            }
            set i 0
            foreach l $luts {
                set s [lindex $order [expr {$i / 8}]]
                lappend pairs $l $s/[lindex {A B C D E F G H} [expr {$i % 8}]]6LUT
                incr i
            }
            foreach s [lrange $order 0 [expr {($i + 7) / 8 - 1}]] { dict unset free $s }
        }
        set ff [get_cells -quiet $c/gb_clk]
        if {[llength $ff]} { lappend pairs $ff [lindex [dict keys $free] 0]/AFF }
        place_cell $pairs
        incr total [expr {[llength $pairs] / 2}]
    }
    puts "placed $total grey-box cells in [expr {([clock milliseconds] - $t0) / 1000.0}] s"
}
"""


def static_tcl(shell_dir, part, odir, n, ranges, threads=16):
    """Static overlay: the shell with the container in its accelerator cell, the tiles as
    reconfigurable partitions (grey boxes) in fabric-only pblocks (whole clock regions include the
    shell's IOBs, HDPR-6), the kernel clock re-routed under DFX rules (Constraints 18-4430)."""
    rp = open(os.path.join(shell_dir, "rp_cell.txt")).read().strip()
    t = [
        "set_param general.maxThreads %d" % threads,
        "set t0 [clock milliseconds]",
        'proc stamp {n} {global t0; puts "STAMP $n [expr ([clock milliseconds] - $t0) / 1000.0]"}',
        "read_verilog %s/container.v" % odir,
        "synth_design -top %s -part %s -mode out_of_context" % (CORE_MODULE, part),
        "write_checkpoint -force %s/container.dcp" % odir,
        "close_design",
        "stamp synth_container",
        "open_checkpoint %s/shell_routed.dcp" % shell_dir,
        "stamp open_shell",
        "read_checkpoint -cell %s %s/container.dcp" % (rp, odir),
        "stamp read_container",
    ]
    for k in range(n):
        t += [
            "set c [get_cells %s/t%d]" % (rp, k),
            "set_property HD.RECONFIGURABLE 1 $c",
            "create_pblock pb_t%d" % k,
            "resize_pblock pb_t%d -add {%s}" % (k, " ".join(ranges[k])),
            "add_cells_to_pblock pb_t%d $c" % k,
            "set_property SNAPPING_MODE ON [get_pblocks pb_t%d]" % k,
        ]
    t += [
        # (the tiles are grey boxes already: container_verilog)
        "stamp partitions",
        "write_checkpoint -force %s/static_parts.dcp" % odir,
        "stamp write_parts",
        PLACE_GREYBOX_TCL,
        "place_greybox %s %d" % (rp, n),
        "stamp place_greybox",
        "set_property IS_ROUTE_FIXED 0 [get_nets -hier -quiet -filter {TYPE == POWER || TYPE == GROUND}]",
        # the shell's locked kernel clock spans only the clock regions of its old loads (Place
        # 30-678 for tiles outside them): unrouted before placement, re-routed under DFX rules
        "set kbuf [get_cells -hier -filter {REF_NAME =~ BUFG* && NAME =~ *clk_kernel*}]",
        "set kclk [get_nets -of [get_pins -of $kbuf -filter {DIRECTION == OUT}]]",
        "set_property IS_ROUTE_FIXED 0 $kclk",
        "route_design -unroute -nets $kclk",
        "place_design",
        "stamp place",
        "route_design",
        "stamp route",
        "report_route_status -file %s/static_route.rpt" % odir,
        "report_timing_summary -file %s/static_timing.rpt" % odir,
        "lock_design -level routing",
        "write_checkpoint -force %s/static_routed.dcp" % odir,
        "stamp write",
        # the full configuration with grey-box tiles (programmed once; the tiles then as partials)
        "write_bitstream -force -no_partial_bitfile %s/static_full.bit" % odir,
        "stamp bitstream",
    ]
    return "\n".join(t) + "\n", rp


def abstract_tcl(odir, rp, ks, threads=4):
    """Abstract shells of the tiles ks (one Vivado session per group, sessions in parallel)."""
    t = ["set_param general.maxThreads %d" % threads,
         "set t0 [clock milliseconds]",
         'proc stamp {n} {global t0; puts "STAMP $n [expr ([clock milliseconds] - $t0) / 1000.0]"}',
         "open_checkpoint %s/static_routed.dcp" % odir, "stamp open"]
    for k in ks:
        t += ["write_abstract_shell -force -cell %s/t%d %s/abs_t%d.dcp" % (rp, k, odir, k), "stamp abs_t%d" % k]
    return "\n".join(t) + "\n"


def build_overlay(shell_dir, part, dev, region, anchor, lib, kn=None, jobs=16):
    """The static overlay of a shell (cached in lib/<key>): static_routed.dcp, abs_t<k>.dcp,
    overlay.json (tiles, pblock ranges, link widths, the accelerator cell). Returns (dir, info)."""
    from concurrent.futures import ThreadPoolExecutor

    kn = kn or overlay_knobs()
    tiles = overlay_tiles(dev, region, kn, anchor)
    key = overlay_key(shell_dir, part, tiles, kn)
    odir = os.path.join(lib, key)
    info_f = os.path.join(odir, "overlay.json")
    if os.path.isfile(info_f):
        info = json.load(open(info_f))
        if all(os.path.isfile(os.path.join(odir, "abs_t%d.dcp" % k)) for k in range(len(tiles))):
            info["status"] = "cached"
            return odir, info
    os.makedirs(odir, exist_ok=True)
    ports = json.load(open(os.path.join(shell_dir, "ports.json")))
    n = len(tiles)
    ranges = [pblock_ranges(dev.sites_in(*tl)) for tl in tiles]
    with open(os.path.join(odir, "container.v"), "w") as f:
        f.write(container_verilog(ports, n, kn))
    tcl, rp = static_tcl(shell_dir, part, odir, n, ranges)
    with open(os.path.join(odir, "static.tcl"), "w") as f:
        f.write(tcl)
    rc, t_static = run_vivado(os.path.join(odir, "static.tcl"), os.path.join(odir, "static.log"), odir)
    info = {"key": key, "tiles": tiles, "ranges": ranges, "kn": kn, "rp": rp, "n": n, "static_s": t_static, "rc": rc}
    if rc != 0 or not os.path.isfile(os.path.join(odir, "static_routed.dcp")):
        info["status"] = "static_failed"
        return odir, info
    groups = [list(range(n))[i::jobs] for i in range(min(jobs, n))]

    def ab(gi):
        tf = os.path.join(odir, "abstract_%d.tcl" % gi)
        with open(tf, "w") as f:
            f.write(abstract_tcl(odir, rp, groups[gi]))
        return run_vivado(tf, os.path.join(odir, "abstract_%d.log" % gi), odir)

    with ThreadPoolExecutor(max_workers=len(groups)) as ex:
        res = list(ex.map(ab, range(len(groups))))
    info["abstract_s"] = max(t for _, t in res)
    ok = all(os.path.isfile(os.path.join(odir, "abs_t%d.dcp" % k)) for k in range(n))
    info["status"] = "built" if ok else "abstract_failed"
    json.dump(info, open(info_f, "w"), indent=1)
    return odir, info


# ---------------------------------------------------------------------------------------------
# per model


def plan_overlay(names, node_res, g, dev, tiles, boundary, extra=None, hard=HARD_UTIL, tile0=True):
    """Nodes to tiles: the nodes with shell ports (IODMAs) into tile 0, the others along the snake
    from tile 0 (tile0=True: the first nodes share tile 0 with the IODMAs while they fit; else from
    tile 1): consecutive nodes share a tile while they fit; the next node goes to the next tile
    that can hold it on its own (tiles in between stay empty and only pass streams through - the
    tiles differ in their DSP / BRAM / URAM columns). A node that fits no tile must be split.
    Tries LUT utilizations 0.5..0.9. Returns ({node: tile}, util) or (None, reason)."""
    from finn.util.rwislands.device import capacity
    from finn.util.rwislands.floorplan import need

    rest = [n for n in names if n not in boundary]
    caps = [capacity(dev.sites_in(*t)) for t in tiles]
    kinds = ("slice", "slicem", "bram", "dsp", "uram")

    def total(mem, t=None):
        tot = {}
        for n in mem + ([b for b in names if b in boundary] if t == 0 else []):
            for k, v in node_res[n].items():
                tot[k] = tot.get(k, 0) + v
        if extra is not None:
            for k, v in extra([rest.index(n) for n in mem if n in rest]).items():
                tot[k] = tot.get(k, 0) + v
        return tot

    last = "the nodes do not fit the overlay's tiles"
    for u in (0.5, 0.6, 0.7, 0.8, 0.9):
        util = {"lut": u, "bram": hard, "dsp": hard, "uram": hard}

        def fits(t, mem):
            req = need(total(mem, t), util)
            return all(caps[t][k] >= req[k] for k in kinds)

        where = {n: 0 for n in names if n in boundary}
        t, mem, ok = (0 if tile0 else 1), [], True
        for n in rest:
            if mem and fits(t, mem + [n]):
                mem.append(n)
                continue
            t2 = t + 1 if mem else t
            while t2 < len(tiles) and not fits(t2, [n]):
                t2 += 1
            if t2 >= len(tiles):
                ok = False
                last = "no tile left for %s at LUT utilization %.1f" % (n, u) if any(
                    fits(x, [n]) for x in range(1, len(tiles))) else "node larger than every tile (split it): %s" % n
                break
            t, mem = t2, [n]
            where[n] = t
            continue
        if ok:
            # (members of a shared tile were appended without an entry: fill them in)
            cur = None
            for n in rest:
                if n in where:
                    cur = where[n]
                else:
                    where[n] = cur
            return where, util
    return None, last


def link_res(w, edges_in=None, pipe=1, kind="rx"):
    """Resources of one link end for the planner: tx = PO output registers; rx = RI input registers
    + the FIFO (distributed RAM, ~1 LUT per 2 bits per 32 words); thru = transit registers."""
    from finn.util.rwislands.links import link_regs, rx_depth

    po, pi, ri, ro, t = link_regs(pipe)
    if kind == "tx":
        return {"ff": (w + 1) * po + pi, "lut": 2}
    if kind == "thru":
        return {"ff": (w + 2) * t}
    lr = ((w + 1) // 2) * max(1, rx_depth(edges_in or 1, pipe) // 32)
    return {"ff": (w + 1) * ri + ro + 16, "lut": lr + 16, "lutram": lr}


def plan_overlay_dp(names, node_res, g, dev, tiles, boundary, hard=HARD_UTIL, tile0=True, kn=None, lut_max=0.9,
                    prefer=None):
    """Nodes to tiles by an exact chain partition (the floorplanning step of RapidStream /
    AutoBridge restricted to the snake): the nodes in graph (topological) order are cut into
    contiguous segments, segment s goes to tile t_s (t_s increasing; tiles in between only pass
    streams through), so that

      * every tile's slice and SLICEM utilization (nodes + link ends + transit registers) is at
        most U <= lut_max, its DSP / BRAM / URAM utilization at most `hard`,
      * every tile edge carries at most wf forward and wb backward bits (data + valid + ready),

    with the smallest U (min-max logic utilization: balanced tiles, so the slowest tile's place and
    route, which grows with its logic cells, is as short as the overlay allows; hard blocks are
    only a capacity limit; cuts at narrow streams follow from the bit limits and the
    link costs). U by bisection, feasibility by a DP over the cut positions (earliest next free
    tile). Returns ({node: tile}, {"U": ..., "lut": ..., ...}) or (None, reason)."""
    import numpy as np

    from finn.util.rwislands.device import capacity
    from finn.util.rwislands.floorplan import FFS_PER_SLICE, LUTS_PER_SLICE

    kn = kn or overlay_knobs()
    pipe = kn.get("pipe", 1)
    wf, wb = kn["wf"], kn["wb"]
    rest = [n for n in names if n not in boundary]
    pos = {n: i for i, n in enumerate(rest)}
    for b in boundary:
        pos[b] = -1
    N, T = len(rest), len(tiles)
    caps = np.array([[c["slice"], c["slicem"], c["dsp"], c["bram"], c["uram"]] for c in
                     (capacity(dev.sites_in(*t)) for t in tiles)], dtype=float)
    caps[caps == 0] = 1e-9

    # streams: (a, b, w) with producer / consumer positions (-1 = tile 0's boundary nodes)
    streams = []
    for n in g:
        for j, (_, w) in enumerate(g[n]["outs"]):
            d = g[n]["dst"].get(j)
            if d is not None and n in pos and d[0] in pos and not (pos[n] == -1 and pos[d[0]] == -1):
                streams.append((pos[n], pos[d[0]], w))
    # bits over the tile edge at cut c (the edge before rest[c]; a stream from a to b crosses the
    # cuts min < c <= max): forward data + valid + the backward streams' ready, and the reverse
    fw = np.zeros(N + 1)
    bw = np.zeros(N + 1)
    for a, b, w in streams:
        lo, hi = min(a, b), max(a, b)
        fwd = b > a
        fw[lo + 1: hi + 1] += (w + 1) if fwd else 1
        bw[lo + 1: hi + 1] += 1 if fwd else (w + 1)
    cut_ok = (fw <= wf) & (bw <= wb)
    ins = {i: [] for i in range(N)}
    outs = {i: [] for i in range(N)}
    for a, b, w in streams:
        if b >= 0:
            ins[b].append((a, w))
        if a >= 0:
            outs[a].append((b, w))

    def node_vec(res):
        lut, ff, carry, lr = res.get("lut", 0), res.get("ff", 0), res.get("carry", 0), res.get("lutram", 0)
        return np.array([lut, ff, carry, lr, res.get("dsp", 0), res.get("bram", 0), res.get("uram", 0)], dtype=float)

    vec = np.array([node_vec(node_res[n]) for n in rest]) if N else np.zeros((0, 7))
    bvec = sum((node_vec(node_res[b]) for b in names if b in boundary), np.zeros(7))
    big = caps.max(axis=0)

    def sites(v):
        # [slice, slicem, dsp, bram, uram] needed by the resource vector v at 100 % utilization
        return np.array([max(v[0] / LUTS_PER_SLICE, v[1] / FFS_PER_SLICE, v[2]), v[3] / LUTS_PER_SLICE, v[4], v[5], v[6]])

    def add(v, r):
        v[0] += r.get("lut", 0)
        v[1] += r.get("ff", 0)
        v[3] += r.get("lutram", 0)

    # every segment [i, j) up to the size of the largest tile: its site needs, nodes + link ends;
    # computed once (the bisection only changes the limits)
    segs = {}
    thru_ff = link_res(0, pipe=pipe, kind="thru")["ff"] // 2  # per stream: (w + 2) * stages
    for i in range(N):
        v = np.zeros(7)
        rx, tx = [], {}
        # streams over cut i, by their far end: transit through [i, j) while far end >= j
        cross = sorted((max(a, b), w) for a, b, w in streams if min(a, b) < i <= max(a, b))
        cw = [0.0] * (len(cross) + 1)
        for q in range(len(cross) - 1, -1, -1):
            cw[q] = cw[q + 1] + cross[q][1] + 2
        q = 0
        for j in range(i + 1, N + 1):
            k = j - 1
            v += vec[k]
            for a, w in ins[k]:
                if a >= i:
                    tx.pop((a, k), None)
                else:
                    rx.append(w)
            for b, w in outs[k]:
                tx[(k, b)] = w
            while q < len(cross) and cross[q][0] < j:
                q += 1
            lv = v.copy()
            for w in tx.values():
                add(lv, link_res(w, pipe=pipe, kind="tx"))
            for w in rx:
                add(lv, link_res(w, edges_in=1, pipe=pipe, kind="rx"))
            lv[1] += cw[q] * thru_ff
            segs[(i, j)] = (sites(lv), sites(lv + bvec))
            if np.any(sites(v) > big):
                break

    def feasible(U):
        lim = np.array([min(U, lut_max)] * 2 + [hard] * 3)
        INF = T + 1
        f = [INF] * (N + 1)
        par = [None] * (N + 1)
        f[0] = 0
        for i in range(N):
            if f[i] >= INF:
                continue
            for j in range(i + 1, N + 1):
                if (i, j) not in segs:
                    break
                s, s0 = segs[(i, j)]
                t = f[i]
                # tile 0 holds the boundary nodes too (and compute nodes only with tile0)
                if t == 0:
                    if i == 0 and tile0 and np.all(s0 <= caps[0] * lim):
                        if 1 < f[j]:
                            f[j], par[j] = 1, (i, 0)
                        continue
                    t = 1
                if not cut_ok[i]:
                    break
                while t < T and not np.all(s <= caps[t] * lim):
                    t += 1
                if t < T and t + 1 < f[j]:
                    f[j], par[j] = t + 1, (i, t)
        if f[N] >= INF:
            reach[0] = max(i for i in range(N + 1) if f[i] < INF)
            return None
        where = {b: 0 for b in names if b in boundary}
        j = N
        while j > 0:
            i, t = par[j]
            for k in range(i, j):
                where[rest[k]] = t
            j = i
        return where

    if N == 0:
        return {b: 0 for b in names if b in boundary}, {"U": 0.0}
    if prefer is not None:
        # incremental build: keep the previous plan if every node has a tile there, the tiles
        # still follow the snake, and every tile and cut is within the limits
        lim = np.array([lut_max] * 2 + [hard] * 3)
        tl = [prefer.get(n) for n in rest]
        ok = all(t is not None for t in tl) and all(a <= b for a, b in zip(tl, tl[1:])) and \
            all(prefer.get(b) == 0 for b in names if b in boundary)
        i = 0
        while ok and i < N:
            j = i
            while j < N and tl[j] == tl[i]:
                j += 1
            seg = segs.get((i, j))
            t = tl[i]
            need = seg[1] if (t == 0 and i == 0) else (seg[0] if seg is not None else None)
            ok = seg is not None and (t > 0 or (i == 0 and tile0)) and np.all(need <= caps[t] * lim) and \
                (cut_ok[i] or (i == 0 and t == 0))
            i = j
        if ok:
            return {n: prefer[n] for n in names}, {"U": None, "kept": True, "lut": lut_max, "dsp": hard, "bram": hard,
                                                   "uram": hard}
    lo, hi = 0.0, 1.0
    reach = [0]
    best = feasible(hi)
    if best is None:
        lim = np.array([lut_max] * 2 + [hard] * 3)
        bad = [rest[i] for i in range(N) if not np.any(np.all(segs[(i, i + 1)][0] <= caps * lim, axis=1))]
        if bad:
            return None, "node larger than every tile (split it): %s" % bad[0]
        r = reach[0]
        need_tot = sites(vec.sum(axis=0) + bvec)
        return None, ("no plan: nodes placed up to %d of %d (%s), cut there %d / %d forward / backward bits "
                      "(limits %d / %d), %d of %d cut positions over the limits; model / tiles: slices %.2f, "
                      "DSP %.2f, BRAM %.2f, URAM %.2f" % (
                          r, N, rest[min(r, N - 1)], fw[r], bw[r], wf, wb, int((~cut_ok).sum()), N + 1,
                          need_tot[0] / caps[:, 0].sum(), need_tot[2] / caps[:, 2].sum(),
                          need_tot[3] / caps[:, 3].sum(), need_tot[4] / caps[:, 4].sum()))
    # (no gain below umin: a tile's place and route has a fixed cost of ~300 s; a small model
    # would otherwise be spread over every tile, one node each)
    umin = kn.get("umin", 35) / 100.0
    w = feasible(umin)
    if w is not None:
        return w, {"U": umin, "lut": umin, "dsp": hard, "bram": hard, "uram": hard}
    lo = umin
    for _ in range(12):
        mid = (lo + hi) / 2
        w = feasible(mid)
        if w is None:
            lo = mid
        else:
            hi, best = mid, w
    return best, {"U": round(hi, 3), "lut": min(hi, lut_max), "dsp": hard, "bram": hard, "uram": hard}


def link_plan(g, where, n_tiles, kn, reserved=None):
    """Bit ranges of the streams between tiles. A stream from tile a to tile b uses every snake edge
    between them: data + valid in the bundle of its direction (forward: away from tile 0), ready in
    the other one. Returns ({(producer, j): [(edge, dir, data_off, ready_off)]}, usage) or raises
    if a bundle overflows."""
    used = {(e, d): 0 for e in range(n_tiles - 1) for d in ("f", "b")}
    # (bits at the start of the bundles of some edges belong to the configuration chain: cfgnet)
    for e, (rf, rb) in (reserved or {}).items():
        used[(e, "f")], used[(e, "b")] = rf, rb
    cap = {"f": kn["wf"], "b": kn["wb"]}
    plan = {}
    for n in g:
        for j, (_, w) in enumerate(g[n]["outs"]):
            dst = g[n]["dst"].get(j)
            if dst is None or where[dst[0]] == where[n]:
                continue
            a, b = where[n], where[dst[0]]
            d, rd = ("f", "b") if b > a else ("b", "f")
            hops = []
            # edges from the producer to the consumer (edge e joins tiles e and e + 1): hops[0]
            # is the producer's, hops[-1] the consumer's
            for e in (range(a, b) if b > a else range(a - 1, b - 1, -1)):
                off, roff = used[(e, d)], used[(e, rd)]
                used[(e, d)] += w + 1
                used[(e, rd)] += 1
                if used[(e, d)] > cap[d] or used[(e, rd)] > cap[rd]:
                    raise RuntimeError("link edge %d (%s) overflows: stream %s.%d of %d bits" % (e, d, n, j, w))
                hops.append((e, d, off, roff))
            plan[(n, j)] = hops
    return plan, used


def cfg_plan(ports, where):
    """The configuration chain (cfgnet) of a model: None if no compute node with an AXI-Lite slave
    lies outside tile 0; else {"n" (all such nodes = funnel ports, tag = index), "A" (AW / AR
    payload bits), "K" (last tile of the chain), "rq" (request bits), "order" [node], "nodes"
    {node: {"tag", "ch": {AW | W | AR | R | B: channel of ports.json}}}, "reserved" {edge: (forward,
    backward) bits}}."""
    from finn.util.rwislands import cfgnet

    ents = [e for e in ports if not e["iodma"]]
    if not ents or max(where[e["node"]] for e in ents) == 0:
        return None
    n = len(ents)
    nodes = {}
    for i, e in enumerate(ents):
        ch = {c["name"].rsplit("_", 1)[1]: c for c in e["in"] + e["out"]}
        assert set(ch) == {"AW", "W", "AR", "R", "B"}, "%s: only AXI-Lite slaves can use the configuration chain" % e["node"]
        nodes[e["node"]] = {"tag": i, "ch": ch}
    a = max(v["ch"]["AW"]["width"] for v in nodes.values())
    K = max(where[e["node"]] for e in ents)
    rf, rb = cfgnet.reserved_bits(a, n)
    return {"n": n, "A": a, "K": K, "rq": cfgnet.req_width(a, n), "order": [e["node"] for e in ents], "nodes": nodes,
            "reserved": {e: (rf, rb) for e in range(K)}}


def _cfg_tile(k, cfg, where, members, pipe):
    """Netlist lines of tile k's part of the configuration chain (cfgnet): the funnel (tile 0) or a
    chain node, their link transmitters / receivers on the reserved bundle bits. Returns (lines,
    module names, {island port name: wire} for the AXI-Lite ports of local nodes)."""
    from finn.util.rwislands import cfgnet
    from finn.util.rwislands.links import rx_depth

    n, a, K, rq = cfg["n"], cfg["A"], cfg["K"], cfg["rq"]
    rw = cfgnet.RESP_W
    t = cfgnet.tag_width(n)
    txq, rxq = "rwi_link_tx_w%d_p%d" % (rq, pipe), "rwi_link_rx_w%d_d%d_p%d" % (rq, rx_depth(1, pipe), pipe)
    txp, rxp = "rwi_link_tx_w%d_p%d" % (rw, pipe), "rwi_link_rx_w%d_d%d_p%d" % (rw, rx_depth(1, pipe), pipe)
    L, mods, isl = [], [], {}
    for x, w in (("q", rq), ("p", rw)):
        for side in ("w", "e"):
            L += ["    wire [%d:0] cfg_%s%s_data;" % (w - 1, x, side), "    wire cfg_%s%s_valid, cfg_%s%s_ready;" % (x, side, x, side)]

    def tx(inst, mod, x, side, out, out_v, rdy):
        L.append("    %s %s (.clk(clk), .rst(rst_q), .s_data(cfg_%s%s_data), .s_valid(cfg_%s%s_valid), "
                 ".s_ready(cfg_%s%s_ready), .l_data(%s), .l_valid(%s), .l_ready(%s));"
                 % (mod, inst, x, side, x, side, x, side, out, out_v, rdy))
        mods.append(mod)

    def rx(inst, mod, x, side, inp, inp_v, rdy):
        L.append("    %s %s (.clk(clk), .rst(rst_q), .l_data(%s), .l_valid(%s), .l_ready(%s), .m_data(cfg_%s%s_data), "
                 ".m_valid(cfg_%s%s_valid), .m_ready(cfg_%s%s_ready));" % (mod, inst, inp, inp_v, rdy, x, side, x, side, x, side))
        mods.append(mod)

    # bundle bits (cfgnet docstring): forward [rq-1:0] request, [rq] its valid, [rq+1] response
    # ready; backward [0] request ready, [rw:1] response, [rw+1] its valid
    if k < K:
        tx("cfg_txq", txq, "q", "e", "e_out[%d:0]" % (rq - 1), "e_out[%d]" % rq, "e_in[0]")
        rx("cfg_rxp", rxp, "p", "e", "e_in[%d:1]" % rw, "e_in[%d]" % (rw + 1), "e_out[%d]" % (rq + 1))
    if k > 0:
        rx("cfg_rxq", rxq, "q", "w", "w_in[%d:0]" % (rq - 1), "w_in[%d]" % rq, "w_out[0]")
        tx("cfg_txp", txp, "p", "w", "w_out[%d:1]" % rw, "w_out[%d]" % (rw + 1), "w_in[%d]" % (rq + 1))
    if k == 0:
        name = cfgnet.funnel_name(n, a, n)
        c = [".clk(clk)", ".rst(rst_q)"]
        for i, nd in enumerate(cfg["order"]):
            ch = cfg["nodes"][nd]["ch"]
            if where[nd] == 0:
                # (a node in tile 0 keeps its direct shell connection; its funnel port idles)
                c += [".s%d_aw_data(%d'b0)" % (i, a), ".s%d_aw_valid(1'b0)" % i, ".s%d_w_data(36'b0)" % i,
                      ".s%d_w_valid(1'b0)" % i, ".s%d_ar_data(%d'b0)" % (i, a), ".s%d_ar_valid(1'b0)" % i,
                      ".s%d_r_ready(1'b0)" % i, ".s%d_b_ready(1'b0)" % i]
                continue
            for x in ("aw", "ar"):
                cc = ch[x.upper()]
                pad = "" if cc["width"] == a else "%d'b0, " % (a - cc["width"])
                c += [".s%d_%s_data({%s%s})" % (i, x, pad, cc["data"]), ".s%d_%s_valid(%s)" % (i, x, cc["valid"]),
                      ".s%d_%s_ready(%s)" % (i, x, cc["ready"])]
            for x in ("w", "r", "b"):
                cc = ch[x.upper()]
                c += [".s%d_%s_data(%s)" % (i, x, cc["data"]), ".s%d_%s_valid(%s)" % (i, x, cc["valid"]),
                      ".s%d_%s_ready(%s)" % (i, x, cc["ready"])]
        c += [".q_data(cfg_qe_data)", ".q_valid(cfg_qe_valid)", ".q_ready(cfg_qe_ready)",
              ".p_data(cfg_pe_data)", ".p_valid(cfg_pe_valid)", ".p_ready(cfg_pe_ready)"]
    else:
        loc = [nd for nd in cfg["order"] if nd in members]
        name = cfgnet.node_name(len(loc), a, n, k < K)
        c = [".clk(clk)", ".rst(rst_q)", ".qi_data(cfg_qw_data)", ".qi_valid(cfg_qw_valid)", ".qi_ready(cfg_qw_ready)",
             ".po_data(cfg_pw_data)", ".po_valid(cfg_pw_valid)", ".po_ready(cfg_pw_ready)"]
        if k < K:
            c += [".qo_data(cfg_qe_data)", ".qo_valid(cfg_qe_valid)", ".qo_ready(cfg_qe_ready)",
                  ".pi_data(cfg_pe_data)", ".pi_valid(cfg_pe_valid)", ".pi_ready(cfg_pe_ready)"]
        if loc:
            c.append(".tags({%s})" % ", ".join("%d'd%d" % (t, cfg["nodes"][nd]["tag"]) for nd in reversed(loc)))
        for j, nd in enumerate(loc):
            ch = cfg["nodes"][nd]["ch"]
            for x in ("aw", "w", "ar", "r", "b"):
                cc = ch[x.upper()]
                for sig in ("data", "valid", "ready"):
                    isl[cc[sig]] = "isl_" + cc[sig]
                    w = cc["width"] if sig == "data" else 1
                    L.append("    wire %s%s;" % ("[%d:0] " % (w - 1) if w > 1 else "", "isl_" + cc[sig]))
                if x in ("aw", "ar") and cc["width"] < a:
                    L += ["    wire [%d:0] cfg_m%d_%s;" % (a - 1, j, x),
                          "    assign isl_%s = cfg_m%d_%s[%d:0];" % (cc["data"], j, x, cc["width"] - 1)]
                    c.append(".m%d_%s_data(cfg_m%d_%s)" % (j, x, j, x))
                else:
                    c.append(".m%d_%s_data(isl_%s)" % (j, x, cc["data"]))
                c += [".m%d_%s_valid(isl_%s)" % (j, x, cc["valid"]), ".m%d_%s_ready(isl_%s)" % (j, x, cc["ready"])]
    L.append("    %s cfg (%s);" % (name, ", ".join(c)))
    mods.append(name)
    return L, mods, isl


def tile_verilog(k, n_tiles, kn, ports, g, members, dcps, links, cfg=None, where=None):
    """tile<k>: the island of its members (netlist.island_verilog, without register slices) plus a
    link receiver per stream entering the tile, a transmitter per stream leaving it, and wires for
    the streams passing through (pipe > 1: through registers). Returns (verilog, link module
    names used: rwi_link_tx_w<W>_p<pipe>, rwi_link_rx_w<W>_d<FIFO depth>_p<pipe>,
    rwi_link_thru_w<W>_s<stages>)."""
    from finn.util.rwislands.links import link_regs, rx_depth
    from finn.util.rwislands.netlist import _in_names, _out_names, island_ports, island_verilog

    body, widths = [], []
    ins, outs = island_ports(g, members)
    mem = set(members)
    wf, wb = kn["wf"], kn["wb"]
    pipe = kn.get("pipe", 1)
    thru = link_regs(pipe)[4]

    def bit(side, d, off, w=1):
        # (west side: edge k-1, east side: edge k; f = forward = west->east)
        name = {("w", "f"): "w_in", ("w", "b"): "w_out", ("e", "f"): "e_out", ("e", "b"): "e_in"}[(side, d)]
        return "%s[%d:%d]" % (name, off + w - 1, off) if w > 1 else "%s[%d]" % (name, off)

    def side_of(edge):
        return "w" if edge == k - 1 else "e"

    for (src, j), hops in links.items():
        w = g[src]["outs"][j][1]
        dstn, i = g[src]["dst"][j]
        for h in range(len(hops)):
            e, d, off, roff = hops[h]
            rd = "b" if d == "f" else "f"
            if e not in (k - 1, k):
                continue
            # this tile is at one end of the hop; is it the producer, the consumer, or a transit?
            prev_hop = hops[h - 1] if h > 0 else None
            if src in mem and h == 0:
                # transmitter: island output -> link
                dname, vname, rname = _out_names(g[src]["id"], j)
                body.append(
                    "    rwi_link_tx_w%d_p%d tx_%s_%d (.clk(clk), .rst(rst_q), .s_data(%s), .s_valid(%s), .s_ready(%s), "
                    ".l_data(%s), .l_valid(%s), .l_ready(%s));"
                    % (w, pipe, g[src]["id"], j, "isl_" + dname, "isl_" + vname, "isl_" + rname,
                       bit(side_of(e), d, off, w), bit(side_of(e), d, off + w), bit(side_of(e), rd, roff)))
                widths.append("rwi_link_tx_w%d_p%d" % (w, pipe))
            elif dstn in mem and h == len(hops) - 1:
                dname, vname, rname = _in_names(g[dstn]["id"], i)
                body.append(
                    "    rwi_link_rx_w%d_d%d_p%d rx_%s_%d (.clk(clk), .rst(rst_q), .l_data(%s), .l_valid(%s), .l_ready(%s), "
                    ".m_data(%s), .m_valid(%s), .m_ready(%s));"
                    % (w, rx_depth(len(hops), pipe), pipe, g[dstn]["id"], i, bit(side_of(e), d, off, w), bit(side_of(e), d, off + w),
                       bit(side_of(e), rd, roff), "isl_" + dname, "isl_" + vname, "isl_" + rname))
                widths.append("rwi_link_rx_w%d_d%d_p%d" % (w, rx_depth(len(hops), pipe), pipe))
            elif prev_hop is not None and prev_hop[0] in (k - 1, k) and e != prev_hop[0]:
                # transit: from the previous hop's bundle (entering) to this hop's bundle (leaving)
                pe, pd, poff, proff = prev_hop
                if thru:
                    body.append("    rwi_link_thru_w%d_s%d thru_%s_%d (.clk(clk), .f_in(%s), .f_out(%s), .r_in(%s), .r_out(%s));"
                                % (w, thru, g[src]["id"], j, bit(side_of(pe), pd, poff, w + 1), bit(side_of(e), d, off, w + 1),
                                   bit(side_of(e), rd, roff), bit(side_of(pe), rd, proff)))
                    widths.append("rwi_link_thru_w%d_s%d" % (w, thru))
                else:
                    body.append("    assign %s = %s;" % (bit(side_of(e), d, off, w + 1), bit(side_of(pe), pd, poff, w + 1)))
                    body.append("    assign %s = %s;" % (bit(side_of(pe), rd, proff), bit(side_of(e), rd, roff)))
    # the reset of the link logic: the island has its own 2-FF pipeline; one more FDRE here
    rst = ["    wire rst_q;", "    FDRE #(.INIT(1'b0)) rst_ff (.C(clk), .CE(1'b1), .R(1'b0), .D(rst), .Q(rst_q));"]
    cfg_isl = {}
    if cfg is not None and k <= cfg["K"]:
        lines, mods, cfg_isl = _cfg_tile(k, cfg, where, members, pipe)
        body += lines
        widths += mods
    if not members:
        # transit only: the streams passing through, no island
        hdr = "module tile%d (\n%s\n);\n" % (k, ",\n".join("    " + p for p in tile_ports(k, n_tiles, kn, ports)))
        stubs = "".join(_link_stub(m) for m in sorted(set(widths)))
        return hdr + "\n".join((rst if cfg_isl or "rst_q" in "\n".join(body) else []) + body + _ties(k, n_tiles, wf, wb, body)) \
            + "\nendmodule\n" + stubs, sorted(set(widths))
    # the island
    isl = island_verilog("island_t%d" % k, g, members, dcps, regslices=False)
    conns = ["        .clk(clk)", "        .rst(rst)"]
    for nn, i, w in ins:
        dname, vname, rname = _in_names(g[nn]["id"], i)
        if dname in cfg_isl:  # AXI-Lite slave on the configuration chain (cfgnet)
            conns += ["        .%s(%s)" % (x, cfg_isl[x]) for x in (dname, vname, rname)]
        elif g[nn]["src"].get(i) is None:  # shell boundary (tile 0)
            conns += ["        .%s(%s)" % (x, x) for x in (dname, vname, rname)]
        else:
            body += ["    wire %s%s;" % (_bus(w), "isl_" + dname), "    wire %s, %s;" % ("isl_" + vname, "isl_" + rname)]
            conns += ["        .%s(%s)" % (x, "isl_" + x) for x in (dname, vname, rname)]
    for nn, j, w in outs:
        dname, vname, rname = _out_names(g[nn]["id"], j)
        if dname in cfg_isl:
            conns += ["        .%s(%s)" % (x, cfg_isl[x]) for x in (dname, vname, rname)]
        elif g[nn]["dst"].get(j) is None:
            conns += ["        .%s(%s)" % (x, x) for x in (dname, vname, rname)]
        else:
            body += ["    wire %s%s;" % (_bus(w), "isl_" + dname), "    wire %s, %s;" % ("isl_" + vname, "isl_" + rname)]
            conns += ["        .%s(%s)" % (x, "isl_" + x) for x in (dname, vname, rname)]
    hdr = "module tile%d (\n%s\n);\n" % (k, ",\n".join("    " + p for p in tile_ports(k, n_tiles, kn, ports)))
    inst = "    island_t%d isl (\n%s\n    );" % (k, ",\n".join(conns))
    v = hdr + "\n".join(rst + body + _ties(k, n_tiles, wf, wb, body)) + "\n" + inst + "\nendmodule\n" + isl
    stubs = "".join(_link_stub(m) for m in sorted(set(widths)))
    return v + stubs, sorted(set(widths))


def _ties(k, n_tiles, wf, wb, body):
    """Undriven bundle outputs tied low (one assign each, so the netlist has no floating outputs)."""
    ties = []
    driven = "\n".join(body)
    for side, d, w in (("w", "b", wb), ("e", "f", wf)):
        if (side == "w" and k == 0) or (side == "e" and k == n_tiles - 1):
            continue
        name = "w_out" if side == "w" else "e_out"
        for b_ in range(w):
            if not _covers(driven, name, b_):
                ties.append("    assign %s[%d] = 1'b0;" % (name, b_))
    return ties


def _covers(text, name, b):
    """Whether bit b of the bundle name is driven somewhere in text (tx/rx ports or assigns)."""
    import re

    for m in re.finditer(r"%s\[(\d+)(?::(\d+))?\]" % name, text):
        hi = int(m.group(1))
        lo = int(m.group(2)) if m.group(2) is not None else hi
        if lo <= b <= hi:
            # only outputs count: the bundle names w_out / e_out are outputs of the tile
            return True
    return False


def _link_mod(name):
    """{"kind": tx | rx | thru, "w", "d" (rx FIFO depth), "p" (pipe), "s" (thru stages)} of a link
    module name."""
    m = re.match(r"rwi_link_(tx|rx|thru)_w(\d+)(?:_d(\d+))?(?:_p(\d+))?(?:_s(\d+))?$", name)
    return {"kind": m.group(1), "w": int(m.group(2)), "d": int(m.group(3) or 0), "p": int(m.group(4) or 1),
            "s": int(m.group(5) or 0)}


def _link_stub(name):
    from finn.util.rwislands import cfgnet

    if cfgnet.parse_name(name):
        return cfgnet.module_stub(name)
    lm = _link_mod(name)
    w = lm["w"]
    if lm["kind"] == "tx":
        return ("(* black_box *)\nmodule %s (input clk, input rst, input [%d:0] s_data, input s_valid, "
                "output s_ready, output [%d:0] l_data, output l_valid, input l_ready);\nendmodule\n" % (name, w - 1, w - 1))
    if lm["kind"] == "thru":
        return ("(* black_box *)\nmodule %s (input clk, input [%d:0] f_in, output [%d:0] f_out, input r_in, "
                "output r_out);\nendmodule\n" % (name, w, w))
    return ("(* black_box *)\nmodule %s (input clk, input rst, input [%d:0] l_data, input l_valid, "
            "output l_ready, output [%d:0] m_data, output m_valid, input m_ready);\nendmodule\n" % (name, w - 1, w - 1))


def link_synth_tcl(mods, odir, part):
    """One session: the link modules `mods` (rwi_link_tx_w<W>_p<P>, rwi_link_rx_w<W>_d<D>_p<P>,
    rwi_link_thru_w<W>_s<S>; out of context, checkpoints in odir)."""
    from finn.util.rwislands.links import LINK_RX, LINK_THRU, LINK_TX, link_regs

    from finn.util.rwislands import cfgnet

    t = ["set_param general.maxThreads 1"]
    for name in mods:
        if cfgnet.parse_name(name):
            # configuration network modules (cfgnet)
            v = os.path.join(odir, name + ".v")
            with open(v, "w") as f:
                f.write(cfgnet.module_verilog(name))
            t += ["read_verilog %s" % v,
                  "synth_design -top %s -part %s -mode out_of_context" % (name, part),
                  "write_checkpoint -force %s" % os.path.join(odir, name + "_synth.dcp"),
                  "close_design", "remove_files -quiet [get_files -quiet]"]
            continue
        lm = _link_mod(name)
        w, d = lm["w"], lm["d"]
        po, pi, ri, ro, _ = link_regs(lm["p"])
        v = os.path.join(odir, name + ".v")
        if lm["kind"] == "tx":
            body = LINK_TX.replace("module rwi_link_tx #(parameter W = 1, parameter PO = 1, parameter PI = 1",
                                   "module %s #(parameter W = %d, parameter PO = %d, parameter PI = %d" % (name, w, po, pi))
        elif lm["kind"] == "thru":
            body = LINK_THRU.replace("module rwi_link_thru #(parameter W = 1, parameter S = 1",
                                     "module %s #(parameter W = %d, parameter S = %d" % (name, w, lm["s"]))
        else:
            body = LINK_RX.replace(
                "module rwi_link_rx #(parameter W = 1, parameter DEPTH = 8, parameter SLACK = 5, parameter RI = 0, parameter RO = 1",
                "module %s #(parameter W = %d, parameter DEPTH = %d, parameter SLACK = %d, parameter RI = %d, parameter RO = %d"
                % (name, w, d, d - 3, ri, ro))
        assert body not in (LINK_TX, LINK_RX, LINK_THRU)
        with open(v, "w") as f:
            f.write(body)
        t += ["read_verilog %s" % v,
              "synth_design -top %s -part %s -mode out_of_context" % (name, part),
              "write_checkpoint -force %s" % os.path.join(odir, name + "_synth.dcp"),
              "close_design", "remove_files -quiet [get_files -quiet]"]
    return "\n".join(t) + "\n"


def tile_tcl(k, odir_tile, tile_v, dcp_files, part, overlay_dir, rp, steps, threads=8):
    """Link the tile's netlist, implement it in the tile's abstract shell, partial bitstream."""
    t = [
        "set_param general.maxThreads %d" % threads,
        "set t0 [clock milliseconds]",
        'proc stamp {n} {global t0; puts "STAMP $n [expr ([clock milliseconds] - $t0) / 1000.0]"}',
        "read_verilog %s" % tile_v,
    ]
    t += ["read_checkpoint %s" % f for f in dcp_files]
    t += [
        "if {[catch {link_design -top tile%d -part %s -mode out_of_context} err]} {" % (k, part),
        "    if {[catch {current_design}] || [llength [get_cells -quiet -hier -filter {IS_BLACKBOX}]] > 0} {error $err}",
        "}",
        "write_checkpoint -force %s/rm.dcp" % odir_tile,
        "stamp link",
        "open_checkpoint %s/abs_t%d.dcp" % (overlay_dir, k),
        "stamp open_abstract",
        "read_checkpoint -cell %s/t%d %s/rm.dcp" % (rp, k, odir_tile),
        "stamp read_rm",
        steps["opt"], "stamp opt",
        steps["place"], "stamp place",
        steps["phys_opt"], "stamp phys_opt",
        steps["route"], "stamp route",
        "report_route_status -file %s/route_status.rpt" % odir_tile,
        "report_timing_summary -file %s/timing_summary.rpt" % odir_tile,
        "report_utilization -file %s/utilization.rpt" % odir_tile,
        "stamp reports",
        "write_bitstream -force -cell %s/t%d %s/tile%d_partial.bit" % (rp, k, odir_tile, k),
        "stamp bitstream",
        # the routed module alone: read into the static design to assemble the full accelerator
        # (verification only; the bitstreams need no assembly)
        "write_checkpoint -force -cell %s/t%d %s/tile%d_rm_routed.dcp" % (rp, k, odir_tile, k),
        "stamp write",
    ]
    return "\n".join(t) + "\n"


def assemble_tcl(overlay_dir, rp, rms, netlist, core_module):
    """Verification: the static overlay with the routed tiles {k: rm_dcp} read into their
    partitions (the others stay grey boxes), the accelerator cell as a functional netlist."""
    # (the static is locked: the grey boxes' GND / VCC / clock routing cannot be removed otherwise,
    # Constraints 18-11729)
    t = ["open_checkpoint %s/static_routed.dcp" % overlay_dir, "lock_design -unlock -level routing"]
    for k, f in sorted(rms.items()):
        # (the static's tiles are grey boxes: read_checkpoint -cell needs a black box, Vivado 12-12243)
        t += ["update_design -cell %s/t%d -black_box" % (rp, k), "read_checkpoint -cell %s/t%d %s" % (rp, k, f)]
    t += ["report_route_status -file %s.route_status.rpt" % netlist,
          "write_verilog -mode funcsim -force -cell %s -rename_top %s %s" % (rp, core_module, netlist)]
    return "\n".join(t) + "\n"


def tile_split_limits(part, kn=None):
    """Resources one part of a split layer may use: `part` percent of a (median) overlay tile's
    usable resources (LUTs at 50 % utilization, DSP / BRAM at `hard`), URAM 60 % of the richest
    tile's; the default 45 % lets two
    parts share a tile (one part per tile left 40 % of the DSPs of the 22-tile overlay empty)."""
    import statistics

    from finn.util.rwislands.device import capacity, load_device
    from finn.util.rwislands.flow import island_region, shell_anchor

    kn = kn or overlay_knobs()
    dev = load_device(part)
    region, first = island_region(part, dev)
    tiles = overlay_tiles(dev, region + list(first), kn, shell_anchor(part, dev))[1:]
    caps = [capacity(dev.sites_in(*t)) for t in tiles]
    med = {k: statistics.median(c[k] for c in caps) for k in ("slice", "dsp", "bram")}
    f, hard = kn["part"] / 100.0, kn["hard"] / 100.0
    # URAM: the URAM-richest tile (URAM weight memories are depth-bound and split by output
    # channel blocks, split_large_mvau.split_mh; ResNet50's last layer needs 125 of a tile's <= 56)
    return {"lut": f * 0.5 * 8 * med["slice"], "dsp": f * hard * med["dsp"], "bram": f * hard * med["bram"],
            "uram": hard * max(c["uram"] for c in caps) * 0.6}
