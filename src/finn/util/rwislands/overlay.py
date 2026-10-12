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

OVERLAY_VERSION = 3
# DSP / BRAM / URAM share of a tile's sites a model may use: the tile pblocks end on clock-region
# rows, and routing containment makes Vivado block the hard blocks at their edges (VGG10 0.25x,
# tile 5 at the device edge: 19 of 54 DSP sites blocked, 16 by place.closeUnroutableSites)
HARD_UTIL = 0.6
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
    crs: clock-region columns per tile."""
    kn = {"cols": 16, "rows": 60, "order": "rows", "wf": 1024, "wb": 256, "boundary_rows": 120, "tiles": 0,
          "gap_cols": 1, "gap_rows": 1, "crs": 1}
    for kv in filter(None, os.environ.get("FINN_RWI_OVERLAY", "").split(",")):
        k, v = kv.split("=")
        kn[k] = v if k == "order" else int(v)
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
    for k in range(n):
        c = ["    .clk(clk)", "    .rst(rst)"]
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


def plan_overlay(names, node_res, g, dev, tiles, boundary, extra=None):
    """Nodes to tiles: the nodes with shell ports (IODMAs) into tile 0, the others along the snake
    from tile 1: consecutive nodes share a tile while they fit; the next node goes to the next tile
    that can hold it on its own (tiles in between stay empty and only pass streams through - the
    tiles differ in their DSP / BRAM / URAM columns). A node that fits no tile must be split.
    Tries LUT utilizations 0.5..0.9. Returns ({node: tile}, util) or (None, reason)."""
    from finn.util.rwislands.device import capacity
    from finn.util.rwislands.floorplan import need

    rest = [n for n in names if n not in boundary]
    caps = [capacity(dev.sites_in(*t)) for t in tiles]
    kinds = ("slice", "slicem", "bram", "dsp", "uram")

    def total(mem):
        tot = {}
        for n in mem:
            for k, v in node_res[n].items():
                tot[k] = tot.get(k, 0) + v
        if extra is not None:
            for k, v in extra([rest.index(n) for n in mem]).items():
                tot[k] = tot.get(k, 0) + v
        return tot

    last = "the nodes do not fit the overlay's tiles"
    for u in (0.5, 0.6, 0.7, 0.8, 0.9):
        util = {"lut": u, "bram": HARD_UTIL, "dsp": HARD_UTIL, "uram": HARD_UTIL}

        def fits(t, mem):
            req = need(total(mem), util)
            return all(caps[t][k] >= req[k] for k in kinds)

        where = {n: 0 for n in names if n in boundary}
        t, mem, ok = 1, [], True
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


def link_plan(g, where, n_tiles, kn):
    """Bit ranges of the streams between tiles. A stream from tile a to tile b uses every snake edge
    between them: data + valid in the bundle of its direction (forward: away from tile 0), ready in
    the other one. Returns ({(producer, j): [(edge, dir, data_off, ready_off)]}, usage) or raises
    if a bundle overflows."""
    used = {(e, d): 0 for e in range(n_tiles - 1) for d in ("f", "b")}
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


def tile_verilog(k, n_tiles, kn, ports, g, members, dcps, links):
    """tile<k>: the island of its members (netlist.island_verilog, without register slices) plus a
    link receiver per stream entering the tile, a transmitter per stream leaving it, and wires for
    the streams passing through. Returns (verilog, link widths used)."""
    from finn.util.rwislands.netlist import _in_names, _out_names, island_ports, island_verilog

    body, widths = [], []
    ins, outs = island_ports(g, members)
    mem = set(members)
    wf, wb = kn["wf"], kn["wb"]
    drive = {}  # output bundle bit ranges -> expression

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
                    "    rwi_link_tx_w%d tx_%s_%d (.clk(clk), .rst(rst_q), .s_data(%s), .s_valid(%s), .s_ready(%s), "
                    ".l_data(%s), .l_valid(%s), .l_ready(%s));"
                    % (w, g[src]["id"], j, "isl_" + dname, "isl_" + vname, "isl_" + rname,
                       bit(side_of(e), d, off, w), bit(side_of(e), d, off + w), bit(side_of(e), rd, roff)))
                widths.append(w)
            elif dstn in mem and h == len(hops) - 1:
                dname, vname, rname = _in_names(g[dstn]["id"], i)
                body.append(
                    "    rwi_link_rx_w%d rx_%s_%d (.clk(clk), .rst(rst_q), .l_data(%s), .l_valid(%s), .l_ready(%s), "
                    ".m_data(%s), .m_valid(%s), .m_ready(%s));"
                    % (w, g[dstn]["id"], i, bit(side_of(e), d, off, w), bit(side_of(e), d, off + w),
                       bit(side_of(e), rd, roff), "isl_" + dname, "isl_" + vname, "isl_" + rname))
                widths.append(w)
            elif prev_hop is not None and prev_hop[0] in (k - 1, k) and e != prev_hop[0]:
                # transit: from the previous hop's bundle (entering) to this hop's bundle (leaving)
                pe, pd, poff, proff = prev_hop
                body.append("    assign %s = %s;" % (bit(side_of(e), d, off, w + 1), bit(side_of(pe), pd, poff, w + 1)))
                body.append("    assign %s = %s;" % (bit(side_of(pe), rd, proff), bit(side_of(e), rd, roff)))
    if not members:
        # transit only: the streams passing through, no island
        hdr = "module tile%d (\n%s\n);\n" % (k, ",\n".join("    " + p for p in tile_ports(k, n_tiles, kn, ports)))
        return hdr + "\n".join(body + _ties(k, n_tiles, wf, wb, body)) + "\nendmodule\n", []
    # the island
    isl = island_verilog("island_t%d" % k, g, members, dcps, regslices=False)
    conns = ["        .clk(clk)", "        .rst(rst)"]
    for nn, i, w in ins:
        dname, vname, rname = _in_names(g[nn]["id"], i)
        if g[nn]["src"].get(i) is None:  # shell boundary (tile 0)
            conns += ["        .%s(%s)" % (x, x) for x in (dname, vname, rname)]
        else:
            body += ["    wire %s%s;" % (_bus(w), "isl_" + dname), "    wire %s, %s;" % ("isl_" + vname, "isl_" + rname)]
            conns += ["        .%s(%s)" % (x, "isl_" + x) for x in (dname, vname, rname)]
    for nn, j, w in outs:
        dname, vname, rname = _out_names(g[nn]["id"], j)
        if g[nn]["dst"].get(j) is None:
            conns += ["        .%s(%s)" % (x, x) for x in (dname, vname, rname)]
        else:
            body += ["    wire %s%s;" % (_bus(w), "isl_" + dname), "    wire %s, %s;" % ("isl_" + vname, "isl_" + rname)]
            conns += ["        .%s(%s)" % (x, "isl_" + x) for x in (dname, vname, rname)]
    hdr = "module tile%d (\n%s\n);\n" % (k, ",\n".join("    " + p for p in tile_ports(k, n_tiles, kn, ports)))
    # the reset of the link logic: the island has its own 2-FF pipeline; one more FDRE here
    rst = ["    wire rst_q;", "    FDRE #(.INIT(1'b0)) rst_ff (.C(clk), .CE(1'b1), .R(1'b0), .D(rst), .Q(rst_q));"]
    inst = "    island_t%d isl (\n%s\n    );" % (k, ",\n".join(conns))
    v = hdr + "\n".join(rst + body + _ties(k, n_tiles, wf, wb, body)) + "\n" + inst + "\nendmodule\n" + isl
    stubs = "".join(_link_stub("tx", wd) + _link_stub("rx", wd) for wd in sorted(set(widths)))
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


def _link_stub(kind, w):
    if kind == "tx":
        return ("(* black_box *)\nmodule rwi_link_tx_w%d (input clk, input rst, input [%d:0] s_data, input s_valid, "
                "output s_ready, output [%d:0] l_data, output l_valid, input l_ready);\nendmodule\n" % (w, w - 1, w - 1))
    return ("(* black_box *)\nmodule rwi_link_rx_w%d (input clk, input rst, input [%d:0] l_data, input l_valid, "
            "output l_ready, output [%d:0] m_data, output m_valid, input m_ready);\nendmodule\n" % (w, w - 1, w - 1))


def link_synth_tcl(widths, odir, part):
    """One session: rwi_link_tx_w<W> / rwi_link_rx_w<W> per width (out of context, checkpoints)."""
    from finn.util.rwislands.links import LINK_RX, LINK_TX

    t = ["set_param general.maxThreads 1"]
    for w in widths:
        for kind, src in (("tx", LINK_TX), ("rx", LINK_RX)):
            name = "rwi_link_%s_w%d" % (kind, w)
            v = os.path.join(odir, name + ".v")
            body = src.replace("module rwi_link_%s #(parameter W = 1" % kind, "module %s #(parameter W = %d" % (name, w))
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
    t = ["open_checkpoint %s/static_routed.dcp" % overlay_dir]
    t += ["read_checkpoint -cell %s/t%d %s" % (rp, k, f) for k, f in sorted(rms.items())]
    t += ["report_route_status -file %s.route_status.rpt" % netlist,
          "write_verilog -mode funcsim -force -cell %s -rename_top %s %s" % (rp, core_module, netlist)]
    return "\n".join(t) + "\n"


def tile_split_limits(part, kn=None):
    """Resources one part of a split layer may use so that it fits a (median) overlay tile: LUTs at
    50 %, DSP / BRAM at 90 % of HARD_UTIL of the tile."""
    import statistics

    from finn.util.rwislands.device import capacity, load_device
    from finn.util.rwislands.flow import island_region, shell_anchor

    kn = kn or overlay_knobs()
    dev = load_device(part)
    region, first = island_region(part, dev)
    tiles = overlay_tiles(dev, region + list(first), kn, shell_anchor(part, dev))[1:]
    caps = [capacity(dev.sites_in(*t)) for t in tiles]
    med = {k: statistics.median(c[k] for c in caps) for k in ("slice", "dsp", "bram")}
    return {"lut": 0.5 * 8 * med["slice"], "dsp": 0.9 * HARD_UTIL * med["dsp"], "bram": 0.9 * HARD_UTIL * med["bram"]}
