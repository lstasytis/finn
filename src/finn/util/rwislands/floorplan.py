# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Islands and their floorplan.

A FINN dataflow graph is a chain (in topological order) of layers connected by AXI streams.
The chain is cut into K islands of consecutive nodes (balanced by synthesized size), and the
islands are laid out as a snake over the accelerator region: the region is split into vertical
lanes, the first lane is filled bottom to top, the next one top to bottom, and so on. Each
island gets the rows of its lane (possibly continuing into the next lane) that hold its
resources at the target utilization. Consecutive islands are neighbours, so the streams between
them are short, and every island is implemented at its final location (no relocation)."""

import math

from finn.util.rwislands.device import capacity, pblock_ranges

# LUTs / FFs per slice (UltraScale+)
LUTS_PER_SLICE = 8
FFS_PER_SLICE = 16


def island_cost(res):
    """Size proxy of a node for balancing the islands (roughly proportional to its place and
    route effort)."""
    return (
        res.get("lut", 0)
        + 0.5 * res.get("ff", 0)
        + 150 * res.get("bram", 0)
        + 100 * res.get("dsp", 0)
        + 400 * res.get("uram", 0)
    )


def partition(costs, k):
    """Cut the sequence of node costs into at most k contiguous segments minimizing the largest
    segment cost. Returns a list of (start, end) index ranges (end exclusive)."""
    n = len(costs)
    k = max(1, min(k, n))
    pre = [0.0]
    for c in costs:
        pre.append(pre[-1] + c)
    inf = float("inf")
    # best[j][i]: minimal max segment cost of the first i nodes in j segments
    best = [[inf] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for j in range(1, k + 1):
        for i in range(1, n + 1):
            for m in range(j - 1, i):
                v = max(best[j - 1][m], pre[i] - pre[m])
                if v < best[j][i]:
                    best[j][i], cut[j][i] = v, m
    # the smallest number of segments reaching the optimum (fewer Vivado runs)
    j = min(range(1, k + 1), key=lambda jj: (round(best[jj][n], 6), jj))
    segs, i = [], n
    while j > 0:
        m = cut[j][i]
        segs.append((m, i))
        i, j = m, j - 1
    return segs[::-1]


def need(res, util):
    """Sites an island with the (summed) resources res needs at the utilization targets."""
    u = util
    slices = max(
        res.get("lut", 0) / LUTS_PER_SLICE,
        res.get("ff", 0) / FFS_PER_SLICE,
        res.get("carry", 0),
    )
    return {
        "slice": math.ceil(slices / u["lut"]),
        "slicem": math.ceil(res.get("lutram", 0) / LUTS_PER_SLICE / u["lut"]),
        "bram": math.ceil(res.get("bram", 0) / u["bram"]),
        "dsp": math.ceil(res.get("dsp", 0) / u["dsp"]),
        "uram": math.ceil(res.get("uram", 0) / u["uram"]),
    }


def make_lanes(dev, x0, x1, n_lanes):
    """Split the tile columns x0..x1 into n_lanes contiguous lanes with similar slice counts;
    lane borders are moved so that every lane has BRAM and DSP columns where possible."""
    cols = [x for x in range(x0, x1 + 1) if x in dev.cols]
    weight = [capacity(dev.cols[x])["slice"] for x in cols]
    total = sum(weight)
    lanes, start, acc = [], 0, 0.0
    for i, w in enumerate(weight):
        acc += w
        if len(lanes) < n_lanes - 1 and acc >= total * (len(lanes) + 1) / n_lanes:
            lanes.append((cols[start], cols[i]))
            start = i + 1
    lanes.append((cols[start], cols[-1]))
    return lanes


def _fits(cap, req):
    return all(cap[k] >= req[k] for k in req)


def allocate(dev, lanes, needs, step=5):
    """Snake allocation. lanes: tile rectangles (x0, x1, y0, y1) in walking order (even lanes
    bottom to top, odd lanes top to bottom). needs: per-island site requirements. Returns, per
    island, a list of tile rectangles (x0, x1, ya, yb), or None if the region is too small."""
    # the snake as a sequence of row bands (lane, ya, yb), in walking order
    bands = []
    for li, (lx0, lx1, y0, y1) in enumerate(lanes):
        rows = list(range(y0, y1 + 1, step))
        if li % 2:
            rows = rows[::-1]
        bands += [(lx0, lx1, r, min(r + step - 1, y1)) for r in rows]
    caps = [capacity(dev.sites_in(*b)) for b in bands]
    res, pos = [], 0
    for req in needs:
        got = dict.fromkeys(req, 0)
        rects = []
        while not _fits(got, req):
            if pos >= len(bands):
                return None
            b, c = bands[pos], caps[pos]
            pos += 1
            for k in got:
                got[k] += c[k]
            # merge with the previous band of the same lane into one rectangle
            if rects and rects[-1][0] == b[0] and (rects[-1][3] + 1 == b[2] or b[3] + 1 == rects[-1][2]):
                r = rects[-1]
                rects[-1] = (r[0], r[1], min(r[2], b[2]), max(r[3], b[3]))
            else:
                rects.append(b)
        res.append(rects)
    return res


def allocate_bestfit(dev, lanes, needs, step=5):
    """Fallback for dense designs: every lane is a stack filled bottom to top, each island
    goes into the lane where its resources fit with the least waste (so islands needing a
    scarce column type, e.g. URAM, end up in the lane that has it). Consecutive islands are no
    longer necessarily adjacent (longer inter-island nets). Returns rects per island or None."""
    bands = []  # per lane: list of (band rect, capacity)
    for lx0, lx1, y0, y1 in lanes:
        rows = list(range(y0, y1 + 1, step))
        bands.append([((lx0, lx1, r, min(r + step - 1, y1)), capacity(dev.sites_in(lx0, lx1, r, min(r + step - 1, y1)))) for r in rows])
    ptr = [0] * len(lanes)
    total = [sum(c["slice"] for _, c in lb) for lb in bands]
    res, prev = [], None
    for req in needs:
        best = None
        for li, lb in enumerate(bands):
            got = dict.fromkeys(req, 0)
            k = ptr[li]
            while k < len(lb) and not _fits(got, req):
                for key in got:
                    got[key] += lb[k][1][key]
                k += 1
            if not _fits(got, req):
                continue
            # waste: slices taken beyond the need (relative), then distance to the previous island
            waste = (got["slice"] - req["slice"]) / max(1, total[li])
            dist = 0 if prev is None else abs(lanes[li][0] - lanes[prev[0]][0])
            key = (round(waste, 3), dist)
            if best is None or key < best[0]:
                best = (key, li, k)
        if best is None:
            return None
        _, li, k = best
        lb = bands[li]
        r0, r1 = lb[ptr[li]][0], lb[k - 1][0]
        res.append([(r0[0], r0[1], r0[2], r1[3])])
        ptr[li] = k
        prev = (li,)
    return res


def floorplan(dev, island_res, region, n_lanes=None, utils=None, first_lanes=()):
    """Place the islands (list of summed resource dicts, in chain order) in the region
    (x0, x1, y0, y1), preceded by the lanes first_lanes ((x0, x1, y0, y1) each, e.g. the
    fabric above the PS). Tries increasing utilization until everything fits. Returns
    (rects per island, pblock ranges per island, utilization used, lanes)."""
    x0, x1, y0, y1 = region
    if n_lanes is None:
        # lanes of about 10 tile columns (a few BRAM/DSP columns each)
        n_lanes = max(1, round((x1 - x0 + 1) / 10))
    lanes = list(first_lanes) + [(a, b, y0, y1) for a, b in make_lanes(dev, x0, x1, n_lanes)]
    for u in utils or (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        # BRAM/DSP/URAM are counted exactly (whole primitives); margin only at low utilization
        util = {"lut": u, "bram": min(1.0, u + 0.4), "dsp": min(1.0, u + 0.4), "uram": 1.0}
        needs = [need(r, util) for r in island_res]
        rects = allocate(dev, lanes, needs)
        if rects is None:
            rects = allocate_bestfit(dev, lanes, needs)
        if rects is not None:
            # one set of ranges per rectangle (an island continuing into the next lane has
            # two rectangles, whose bounding box would overlap other islands)
            ranges = [[g for r in rs for g in pblock_ranges(dev.sites_in(*r))] for rs in rects]
            return rects, ranges, util, lanes
    raise RuntimeError("the islands do not fit into the region %s" % (region,))
