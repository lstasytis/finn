# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Site map of an FPGA part for the island floorplanner.

One Vivado run per part (cached in $FINN_BUILD_DIR/rwislands/devices) lists every SLICE,
BRAM, DSP and URAM site with its tile coordinates. Tile columns (x) and tile rows (y) form one
grid for all site types: a CLB tile is one row high, BRAM/DSP/URAM tiles span 5 rows (their y
is the lowest of those rows, a multiple of 5 in UltraScale+ parts)."""

import os
import re
from collections import defaultdict

from finn.util.dynarapid.tools import run_vivado

# resource kinds and the site types that provide them
KINDS = ("slice", "slicem", "bram", "dsp", "uram")

_DUMP_TCL = r"""
link_design -part %s
set f [open %s w]
foreach s [get_sites -filter {SITE_TYPE =~ SLICE* || SITE_TYPE =~ RAMB* || SITE_TYPE =~ DSP48* || SITE_TYPE == URAM288}] {
  regexp {_X(\d+)Y(\d+)$} [get_tiles -of $s] -> tx ty
  puts $f "$s [get_property SITE_TYPE $s] [get_property CLOCK_REGION $s] $tx $ty"
}
close $f
"""


class Site:
    __slots__ = ("name", "type", "cr", "x", "y", "prefix", "sx", "sy")

    def __init__(self, name, type_, cr, x, y):
        self.name, self.type, self.cr, self.x, self.y = name, type_, cr, int(x), int(y)
        m = re.match(r"(\w+?)_X(\d+)Y(\d+)$", name)
        self.prefix, self.sx, self.sy = m.group(1), int(m.group(2)), int(m.group(3))


class Device:
    def __init__(self, part, sites):
        self.part = part
        self.sites = sites
        self.cols = defaultdict(list)  # tile x -> sites
        for s in sites:
            self.cols[s.x].append(s)
        self.xmax = max(self.cols)
        self.ymax = max(s.y for s in sites)

    def sites_in(self, x0, x1, y0, y1):
        """Sites of the tile rectangle [x0, x1] x [y0, y1] (inclusive)."""
        res = []
        for x in range(x0, x1 + 1):
            res += [s for s in self.cols.get(x, ()) if y0 <= s.y <= y1]
        return res


def capacity(sites):
    """Resources of a site list: {kind: count} (bram in RAMB36 tiles)."""
    cap = dict.fromkeys(KINDS, 0)
    for s in sites:
        if s.type.startswith("SLICE"):
            cap["slice"] += 1
            if s.type == "SLICEM":
                cap["slicem"] += 1
        elif s.prefix == "RAMB36":
            cap["bram"] += 1
        elif s.type.startswith("DSP"):
            cap["dsp"] += 1
        elif s.type == "URAM288":
            cap["uram"] += 1
    return cap


def pblock_ranges(sites):
    """Vivado pblock ranges covering exactly the given sites (no other site of the device):
    per site name prefix, per site column the runs of consecutive site rows, merged across
    adjacent site columns with identical runs. For a full tile rectangle this is one range per
    prefix and contiguous column block; for a rectangle with notches (sites excluded, e.g. by
    a reconfigurable partition's snapping) the notches stay out."""
    by = defaultdict(lambda: defaultdict(list))
    for s in sites:
        by[s.prefix][s.sx].append(s.sy)
    ranges = []
    for p in sorted(by):
        runs = {}
        for sx, ys in by[p].items():
            ys = sorted(set(ys))
            rr, start = [], ys[0]
            for a, b in zip(ys, ys[1:] + [None]):
                if b != a + 1:
                    rr.append((start, a))
                    start = b
            runs[sx] = tuple(rr)
        cols = sorted(runs)
        i = 0
        while i < len(cols):
            j = i
            while j + 1 < len(cols) and cols[j + 1] == cols[j] + 1 and runs[cols[j + 1]] == runs[cols[i]]:
                j += 1
            for y0, y1 in runs[cols[i]]:
                ranges.append("%s_X%dY%d:%s_X%dY%d" % (p, cols[i], y0, p, cols[j], y1))
            i = j + 1
    return ranges


_LAGUNA_TCL = r"""
link_design -part %s
set f [open %s w]
foreach s [get_sites -quiet LAGUNA*] {
  regexp {_X(\d+)Y(\d+)$} [get_tiles -of $s] -> tx ty
  puts $f "$s LAGUNA [get_property CLOCK_REGION $s] $tx $ty"
}
close $f
"""


def load_laguna(part, cache_dir=None):
    """LAGUNA sites (SLR crossings) of a part, same format as the site map; separate from the
    floorplanner's site map (a reconfigurable partition that spans an SLR boundary needs them in
    its pblock)."""
    cache_dir = cache_dir or os.path.join(os.environ["FINN_BUILD_DIR"], "rwislands", "devices")
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, part + ".laguna")
    if not os.path.isfile(f):
        tcl = os.path.join(cache_dir, part + "_laguna.tcl")
        with open(tcl, "w") as t:
            t.write(_LAGUNA_TCL % (part, f + ".tmp"))
        rc, _ = run_vivado(tcl, os.path.join(cache_dir, part + "_laguna.log"), cache_dir)
        assert rc == 0 and os.path.isfile(f + ".tmp"), "LAGUNA dump of %s failed" % part
        os.replace(f + ".tmp", f)
    with open(f) as fh:
        return [Site(*v) for v in (line.split() for line in fh) if len(v) == 5]


def load_device(part, cache_dir=None):
    cache_dir = cache_dir or os.path.join(os.environ["FINN_BUILD_DIR"], "rwislands", "devices")
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, part + ".sites")
    if not os.path.isfile(f) or os.path.getsize(f) == 0:
        tcl = os.path.join(cache_dir, part + ".tcl")
        with open(tcl, "w") as t:
            t.write(_DUMP_TCL % (part, f + ".tmp"))
        rc, _ = run_vivado(tcl, os.path.join(cache_dir, part + ".log"), cache_dir)
        assert rc == 0 and os.path.isfile(f + ".tmp"), "site dump of %s failed" % part
        os.replace(f + ".tmp", f)
    sites = []
    with open(f) as fh:
        for line in fh:
            v = line.split()
            if len(v) == 5:
                sites.append(Site(*v))
    return Device(part, sites)
