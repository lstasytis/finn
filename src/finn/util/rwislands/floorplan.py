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


def class_partition(names, res, costs, k, k_scarce=2):
    """Islands for dense designs: nodes using a scarce column resource (URAM) form their own
    islands (k_scarce of them, chain order among themselves), placed first; the other nodes are
    cut into k chain-order islands. Returns member lists (scarce islands first)."""
    scarce = [i for i, n in enumerate(names) if res[n].get("uram", 0) > 0]
    rest = [i for i in range(len(names)) if i not in set(scarce)]
    groups = []
    for idx, kk in ((scarce, k_scarce), (rest, k)):
        if not idx:
            continue
        for a, b in partition([costs[i] for i in idx], kk):
            groups.append([names[i] for i in idx[a:b]])
    return groups


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
    bands, lane_of = [], []
    for li, (lx0, lx1, y0, y1) in enumerate(lanes):
        rows = list(range(y0, y1 + 1, step))
        if li % 2:
            rows = rows[::-1]
        bands += [(lx0, lx1, r, min(r + step - 1, y1)) for r in rows]
        lane_of += [li] * len(rows)
    caps = [capacity(dev.sites_in(*b)) for b in bands]

    def adjacent(la, lb):
        # an island may only continue into a lane that touches its current one (a pblock
        # with CONTAIN_ROUTING must be connected, e.g. not across the shell or an SLR gap)
        a, b = lanes[la], lanes[lb]
        x_touch = a[1] + 1 >= b[0] and b[1] + 1 >= a[0]
        y_touch = a[3] + 1 >= b[2] and b[3] + 1 >= a[2]
        return x_touch and y_touch

    res, pos = [], 0
    for req in needs:
        got = dict.fromkeys(req, 0)
        rects = []
        while not _fits(got, req):
            if pos >= len(bands):
                return None
            if rects and lane_of[pos] != lane_of[pos - 1] and not adjacent(lane_of[pos - 1], lane_of[pos]):
                # restart the island in the next lane (the rest of the previous lane stays unused)
                got = dict.fromkeys(req, 0)
                rects = []
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
    cap_all = {key: max(1, sum(c[key] for lb in bands for _, c in lb)) for key in needs[0]}
    # hardest islands first: largest share of the scarcest resource they need
    order = sorted(
        range(len(needs)),
        key=lambda i: -max(needs[i][key] / cap_all[key] for key in ("bram", "uram", "dsp")),
    )
    res, prev = [None] * len(needs), None
    for i in order:
        req = needs[i]
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
        res[i] = [(r0[0], r0[1], r0[2], r1[3])]
        ptr[li] = k
        prev = (li,)
    return res


def allocate_rects(dev, needs, allowed, step=5):
    """2D rectangle packing for dense designs: islands (hardest first: largest share of the
    scarcest resource) each get the free rectangle of the tile grid (any width and position
    within the allowed rectangles) that covers their needs with the least waste, waste being
    the resources taken beyond the need, weighted by scarcity. Returns rects per island or None."""
    import numpy as np

    xs0 = min(a[0] for a in allowed)
    xs1 = max(a[1] for a in allowed)
    ymax = max(a[3] for a in allowed)
    nx, nb = xs1 - xs0 + 1, ymax // step + 1
    kinds = list(needs[0])
    cap = np.zeros((len(kinds), nx, nb))
    for st in dev.sites:
        if xs0 <= st.x <= xs1:
            c = capacity([st])
            for ki, k in enumerate(kinds):
                cap[ki, st.x - xs0, st.y // step] += c[k]
    occ = np.ones((nx, nb), dtype=np.int64)  # 1 = not usable
    for x0, x1, y0, y1 in allowed:
        occ[x0 - xs0 : x1 - xs0 + 1, y0 // step : y1 // step + 1] = 0
    # prefix sums (padded) for O(1) rectangle sums
    pc = np.zeros((len(kinds), nx + 1, nb + 1))
    pc[:, 1:, 1:] = cap.cumsum(1).cumsum(2)
    total = np.maximum(1, cap.sum(axis=(1, 2)))
    order = sorted(
        range(len(needs)),
        key=lambda i: -max(needs[i][k] / total[kinds.index(k)] for k in ("bram", "uram", "dsp", "slice")),
    )
    res = [None] * len(needs)
    for i in order:
        req = np.array([needs[i][k] for k in kinds], dtype=float)
        po = np.zeros((nx + 1, nb + 1), dtype=np.int64)
        po[1:, 1:] = occ.cumsum(0).cumsum(1)
        best = None
        for a in range(nx):
            for b in range(a, nx):
                # capacity of columns a..b per band, cumulative over bands
                colsum = pc[:, b + 1, :] - pc[:, a, :]  # (kinds, nb+1) prefix over bands
                if np.any(colsum[:, -1] < req):
                    continue
                occ_col = po[b + 1, :] - po[a, :]  # prefix over bands of occupied cells
                for y0 in range(nb):
                    # smallest y1 with enough capacity (bisect on the prefix)
                    lo, hi = y0 + 1, nb
                    if np.any(colsum[:, hi] - colsum[:, y0] < req):
                        continue
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if np.all(colsum[:, mid] - colsum[:, y0] >= req):
                            hi = mid
                        else:
                            lo = mid + 1
                    y1 = lo
                    if occ_col[y1] - occ_col[y0] > 0:
                        continue
                    got = colsum[:, y1] - colsum[:, y0]
                    waste = float(np.sum((got - req) / total))
                    if best is None or waste < best[0]:
                        best = (waste, a, b, y0, y1)
        if best is None:
            return None
        _, a, b, y0, y1 = best
        occ[a : b + 1, y0:y1] = 1
        res[i] = [(a + xs0, b + xs0, y0 * step, y1 * step - 1)]
    return res


def floorplan(dev, island_res, region, n_lanes=None, utils=None, first_lanes=()):
    """Place the islands (list of summed resource dicts, in chain order) in the region
    (x0, x1, y0, y1), preceded by the lanes first_lanes ((x0, x1, y0, y1) each, e.g. the
    fabric above the PS). Tries increasing utilization until everything fits. Returns
    (rects per island, pblock ranges per island, utilization used, lanes)."""
    # region: one rectangle (x0, x1, y0, y1) or a list of them (e.g. one per SLR), each cut
    # into lanes of about 10 tile columns (a few BRAM/DSP columns each)
    lanes = list(first_lanes)
    for x0, x1, y0, y1 in region if isinstance(region, list) else [region]:
        nl = n_lanes or max(1, round((x1 - x0 + 1) / 10))
        lanes += [(a, b, y0, y1) for a, b in make_lanes(dev, x0, x1, nl)]
    # the snake (consecutive islands adjacent) at any utilization before the 2D packing
    tries = [(u, alloc) for alloc in ("snake", "rects") for u in (utils or (0.5, 0.6, 0.7, 0.8, 0.9, 0.95))]
    for u, alloc in tries:
        # BRAM/DSP/URAM are counted exactly (whole primitives); margin only at low utilization
        util = {"lut": u, "bram": min(1.0, u + 0.4), "dsp": min(1.0, u + 0.4), "uram": 1.0}
        needs = [need(r, util) for r in island_res]
        if alloc == "snake":
            rects = allocate(dev, lanes, needs)
        else:
            rects = allocate_rects(dev, needs, list(lanes))
        if rects is not None:
            # one set of ranges per rectangle (an island continuing into the next lane has
            # two rectangles, whose bounding box would overlap other islands)
            ranges = [[g for r in rs for g in pblock_ranges(dev.sites_in(*r))] for rs in rects]
            return rects, ranges, util, lanes
    raise RuntimeError("the islands do not fit into the region %s" % (region,))
