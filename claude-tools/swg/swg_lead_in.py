# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The lead-in block and the frame's read budget, sized by measured occupancy.

A nest that lumps the frame's wait at the head has to say what happens in those
cycles. The counts are settled by two facts, measured on all 384 configurations
rather than assumed (``check`` re-runs them):

1. **Every write-idle cycle of a frame carries a read.** 0 exceptions.
2. **Therefore ``windup <= LAST_READ_ELEM + 1``** -- the idle cycles are a subset
   of the cycles that read. So a cap of ``min(windup, n_read)`` is a no-op, and
   an over-draw is never the lead-in claiming too many reads. ``check --budget``
   shows where it is instead: the rows' draws come from ``TAIL_INCR_W``/``_H``
   and do not know about ``windup``, so a bigger lead-in leaves
   ``budget - n_first - (h-2)*n_mid`` negative and the rows place words the
   frame does not have. Six configurations on the models tier, thirteen overall.

But the counts are not the whole shape, and the shape is what a FIFO sizer reads.
**The lead-in does not carry ``windup`` reads.** It carries as many as leave the
read stream exactly ``peak`` ahead of the beat stream when the rows begin, where
``peak`` is the solver's measured peak occupancy. Those coincide only when the
whole wait really is at the head; on a stride-2 depthwise window it is spread
through the frame and ``windup`` overstates the head burst by up to 249%, which
is a wrong peak occupancy and a sizer that stops trading depth away. Several
other shapes were understated instead, two of them by 100%.

Occupancy is non-increasing through the rows -- every row cycle emits a beat and
reads at most one word -- so the frame's peak *is* the value at the end of the
lead-in, and setting it there sets it exactly.

``lead_in_block(cycles, reads, writes)``
    the run-length block, exact on all three counts. Both streams are spread on
    **one** cycle grid, so neither is placed assuming the other is dense.
``lead_in_node(name, cycles, reads, writes)``
    the same, as a ``Characteristic_Node`` nest: a segment per borrowed beat,
    each two repeated children. 44 entries where the flat form costs 34830.
``allocate(budget, weights, cap)``
    the rows' share, water-filled to the budget instead of to the draws.
``swg_frame_reads(p, windup, peak, ...)``
    the split, summing to ``n_read`` exactly for any ``windup`` the solver gives.
``compose(p, windup, peak, ...)``
    the composed frame's cumulative curves, so the occupancy the composition
    produces can be scored against the solver's -- the check that was missing
    when the lead-in was validated on counts alone.

    from swg_coupled import swg_wait
    from swg_lead_in import swg_frame_reads, lead_in_node
    w = swg_wait(p, style)
    share = swg_frame_reads(p, w["total_gap"], w["peak"], epw=epw, beats=w["beats"])
    node = lead_in_node("lead_in", share["cycles"], share["lead_in"], share["paced"])
    # and use share["paced"] for the give-back, or the writes stop balancing

    python claude-tools/swg/swg_lead_in.py check --matrix all
    python claude-tools/swg/swg_lead_in.py check --budget --matrix models
"""

import argparse
import numpy as np
import os
import sys

os.environ.setdefault(
    "FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swg_fsm import swg_params  # noqa: E402


def _late(cycles, count):
    """A 0/1 mask of ``count`` events over ``cycles``, each as late as its share.

    ``floor`` boundaries, so the first event closes the first share rather than
    opening it. Exact by construction: the mask sums to ``count`` because it is
    a difference of a monotone staircase from 0 to ``count``.
    """
    if cycles <= 0 or count <= 0:
        return np.zeros(max(0, cycles), dtype=np.int64)
    edge = (np.arange(cycles + 1, dtype=np.int64) * count) // cycles
    return np.diff(edge)


def _early(cycles, count):
    """The same, each event as early as its share -- ``ceil`` boundaries."""
    if cycles <= 0 or count <= 0:
        return np.zeros(max(0, cycles), dtype=np.int64)
    edge = -((-np.arange(cycles + 1, dtype=np.int64) * count) // cycles)
    return np.diff(edge)


def lead_in_block(cycles, reads, writes):
    """``[(run, [read, write]), ...]`` of exactly ``cycles``, ``reads``, ``writes``.

    Both streams are laid on the same grid of ``cycles`` and then run-length
    encoded together, which is the whole point: the writes are not placed on the
    assumption that every cycle also reads, nor the reads on the assumption that
    the writes are somewhere else. Each is exact on its own count, and they
    interleave wherever they land.

    Reads take the late edge of their share and writes the early edge, so where
    the block cannot honour both it reads later and writes sooner than an even
    split -- the direction that leaves a FIFO deeper rather than shallower.
    """
    cycles = int(max(0, cycles))
    if cycles == 0:
        return []
    reads = int(min(max(0, reads), cycles))
    writes = int(min(max(0, writes), cycles))
    grid = np.stack([_late(cycles, reads), _early(cycles, writes)], axis=1)
    cut = np.flatnonzero(np.any(np.diff(grid, axis=0) != 0, axis=1)) + 1
    start = np.concatenate(([0], cut))
    run = np.diff(np.concatenate((start, [cycles])))
    return [(int(n), [int(v[0]), int(v[1])]) for n, v in zip(run, grid[start])]


def _spread_node(name, cycles, reads, writes):
    """One segment: its beats first, then its reads spread over what is left.

    Exact on all three counts for any combination, including more writes than
    the segment has room to also read on. The reads are grouped rather than
    interleaved -- longer gaps first, so a read lands no earlier than an even
    spread would put it -- which makes the segment two repeated children instead
    of one run per read. Only the arrangement *inside* the segment changes; the
    frame's peak is the value at the end of the lead-in, which is unchanged.
    """
    from finn.util.basic import Characteristic_Node

    def leaf(runs):
        return Characteristic_Node(name, [(int(n), list(v)) for n, v in runs if n > 0], True)

    cycles = max(0, int(cycles))
    writes = min(max(0, int(writes)), cycles)
    reads = min(max(0, int(reads)), cycles)
    body = cycles - writes
    # reads that cannot fit after the beats have to share a cycle with one
    shared = max(0, reads - body)
    left = reads - shared
    lead = [(writes - shared, [0, 1]), (shared, [1, 1])]
    if left <= 0 or body <= 0:
        return leaf(lead + [(body, [0, 0])])
    if left >= body:
        return leaf(lead + [(body, [1, 0])])
    long_gap, few = divmod(body - left, left)
    kids = [(1, leaf(lead))] if writes else []
    for count, gap in ((few, long_gap + 1), (left - few, long_gap)):
        if count > 0:
            kids.append((count, leaf([(gap, [0, 0]), (1, [1, 0])])))
    return Characteristic_Node(name, kids, False)


def lead_in_node(name, cycles, reads, writes):
    """The lead-in as a ``Characteristic_Node``, built as a nest rather than a leaf.

    Exact on ``(cycles, reads, writes)`` and on the occupancy it leaves behind,
    which is ``reads - writes`` and is the frame's peak. A flat encoding of a
    sparse lead-in is one run per read -- 34830 of them on the widest depthwise
    window; this is a segment per borrowed beat, each two repeated children, so
    tens of entries instead of tens of thousands.
    """
    from finn.util.basic import Characteristic_Node

    cycles = max(0, int(cycles))
    reads = min(max(0, int(reads)), cycles)
    writes = min(max(0, int(writes)), cycles)
    if cycles == 0:
        return Characteristic_Node(name, [], True)
    segments = min(writes + 1, cycles)
    edge = [(i * cycles) // segments for i in range(segments + 1)]
    spans = [edge[i + 1] - edge[i] for i in range(segments)]
    # a segment holds no more of either stream than it has cycles, so both are
    # shared out against that cap; an even cut hands a short segment more reads
    # than it can carry and they are then silently dropped
    rds, wrs = _share(reads, spans), _share(writes, spans)
    kids = [
        (1, _spread_node(name, spans[i], rds[i], wrs[i])) for i in range(segments) if spans[i] > 0
    ]
    return Characteristic_Node(name, kids, False)


def _share(total, caps):
    """Split ``total`` over slots, none above its own cap. Sums exactly."""
    out = [0] * len(caps)
    room = sum(caps)
    total = min(max(0, int(total)), room)
    if room <= 0 or total == 0:
        return out
    for i, c in enumerate(caps):
        out[i] = min(c, c * total // room)
    short, i = total - sum(out), 0
    while short > 0:
        give = min(caps[i] - out[i], short)
        out[i] += give
        short -= give
        i = (i + 1) % len(caps)
    return out


def allocate(budget, weights, cap):
    """Split ``budget`` over rows in the proportions ``weights``, none above ``cap``.

    Sums to ``min(budget, len(weights) * cap)`` exactly and holds every row
    inside ``cap`` -- water-filling, so a row that would overflow spills to the
    rows that still have room instead of being clipped away. The shortfall when
    even that cannot fit is what the caller's drain block takes; it is returned
    by comparing the sum, never dropped silently.

    Where the weights already fit they are kept as they are. Where they do not
    they are scaled, and the rounding remainder goes to the earliest rows, which
    is where the buffer's own head start puts it.
    """
    weights = [max(0, int(x)) for x in weights]
    rows, cap = len(weights), max(0, int(cap))
    total = min(max(0, int(budget)), rows * cap)
    if rows == 0 or total == 0:
        return [0] * rows
    scale = sum(weights)
    if scale <= total:
        out = [min(cap, w) for w in weights]
    else:
        out = [min(cap, w * total // scale) for w in weights]
    short = total - sum(out)
    i = 0
    while short > 0:  # terminates: total <= rows*cap, so room always remains
        give = min(cap - out[i], short)
        out[i] += give
        short -= give
        i = (i + 1) % rows
    return out


def swg_frame_reads(p, windup, peak, rows=None, per_row=None, cap=None, epw=None, beats=None):
    """How a frame's ``n_read`` words divide between the lead-in and the rows.

    **The lead-in does not take ``windup`` reads.** It takes as many as put the
    read stream exactly ``peak`` ahead of the beat stream at the moment the rows
    begin, and ``peak`` is the solver's measured peak occupancy, not the frame's
    idle-cycle count. Those are the same number only when the whole wait really
    is at the head of the frame; on a stride-2 depthwise window the wait is
    spread through it and ``windup`` overstates the head burst by up to 29%,
    which is a 51% error in peak occupancy and the reason a sizer that was
    trading depth away stops doing it.

    Occupancy is non-increasing through the rows -- every row cycle emits a beat
    and reads at most one word -- so the frame's peak *is* the value at the end
    of the lead-in, and setting it there sets it exactly.

    The borrowed beats give way to that: ``paced`` is capped at what the lead-in
    has room for once its reads are placed, because a beat inside the lead-in
    cancels one of them. The caller must use the ``paced`` returned here for the
    give-back too, or the write count stops balancing.
    """
    n_read = p["LAST_READ_ELEM"] + 1
    # the parallel style has no window buffer, so no beats to borrow
    epw = p.get("ELEM_PER_WINDOW", 1) if epw is None else epw
    windup = int(max(0, windup))
    peak = int(_clip(peak, 0, min(windup, n_read)))
    room = max(0, min(windup, n_read) - peak)
    paced = int(_clip(min(epw - 1, windup if beats is None else beats - 1), 0, room))
    lead = peak + paced
    budget = n_read - lead
    rows = 0 if rows is None else int(rows)
    weights = [budget] * rows if per_row is None else list(per_row)
    cap = budget if cap is None else int(cap)
    share = allocate(budget, weights, cap)
    return dict(
        lead_in=lead,
        paced=paced,
        peak=peak,
        cycles=windup,
        budget=budget,
        rows=share,
        drain=budget - sum(share),
        n_read=n_read,
    )


def _clip(v, lo, hi):
    return max(lo, min(int(v), hi))


def compose(p, windup, peak, epw=None, beats=None):
    """The composed frame's cumulative (reads, writes), as the nest emits them.

    Lead-in, then the beats at one per cycle carrying what reads are left, then
    the drain. This is the shape ``swg_default_nest`` builds; it is here so the
    occupancy the composition produces can be compared against the solver's own,
    which is the check that was missing when the lead-in was validated on counts
    alone.
    """
    beats = p["LAST_WRITE_ELEM"] + 1 if beats is None else int(beats)
    share = swg_frame_reads(p, windup, peak, epw=epw, beats=beats)
    head = lead_in_node("lead_in", share["cycles"], share["lead_in"], share["paced"]).deltas()
    body = np.zeros((beats - share["paced"], 2), dtype=np.int64)
    body[:, 1] = 1
    body[:, 0] = _late(body.shape[0], min(share["budget"], body.shape[0]))
    tail = np.zeros((max(0, share["budget"] - int(body[:, 0].sum())), 2), dtype=np.int64)
    tail[:, 0] = 1
    frame = np.concatenate([head, body, tail]) if len(head) else np.concatenate([body, tail])
    return np.cumsum(frame, axis=0)


def _iter(matrix):
    from swg_configs import get_matrix
    from swg_tav import build_swg

    for c in get_matrix(matrix):
        try:
            _, inst = build_swg(c)
        except Exception:
            continue
        if "_rtl" not in type(inst).__name__:
            continue
        style = inst.select_impl_style()
        p = swg_params(inst, style)
        if p is not None:
            yield c, inst, style, p


def cmd_check(args):
    """The two facts, the block's counts, and the composed frame's occupancy."""
    import swg_coupled as SC

    n = idle_bad = block_bad = short = fold_bad = 0
    entries = flat_entries = 0
    worst_peak = worst_read = worst_write = 0
    rows = []
    for c, inst, style, p in _iter(args.matrix):
        w = SC.swg_wait(p, style)
        d = SC.delta(p, style)
        if w is None or d is None:
            continue
        n += 1
        idle = d[:, 1] == 0
        windup = int(idle.sum())
        if int(d[idle, 0].sum()) != windup or windup != w["total_gap"]:
            idle_bad += 1
        beats = w["beats"]
        share = swg_frame_reads(p, windup, w["peak"], epw=p.get("ELEM_PER_WINDOW", 1), beats=beats)
        block = lead_in_block(share["cycles"], share["lead_in"], share["paced"])
        got = (
            sum(r for r, _ in block),
            sum(r * v[0] for r, v in block),
            sum(r * v[1] for r, v in block),
        )
        if got != (share["cycles"], share["lead_in"], share["paced"]):
            block_bad += 1
        node = lead_in_node("lead_in", share["cycles"], share["lead_in"], share["paced"])
        made = node.deltas()
        if (
            made.shape[0] != share["cycles"]
            or int(made[:, 0].sum()) != share["lead_in"]
            or int(made[:, 1].sum()) != share["paced"]
        ):
            fold_bad += 1
        entries = max(entries, _entries(node))
        flat_entries = max(flat_entries, len(block))
        if share["lead_in"] + sum(share["rows"]) + share["drain"] != share["n_read"]:
            short += 1
        ref = np.cumsum(d, axis=0)
        cand = compose(p, windup, w["peak"], epw=p.get("ELEM_PER_WINDOW", 1), beats=beats)
        m = min(len(ref), len(cand))
        peak_ref = int((ref[:, 0] - ref[:, 1]).max())
        peak_cand = int((cand[:, 0] - cand[:, 1]).max())
        e_peak = peak_cand - peak_ref
        e_read = int(np.abs(cand[:m, 0] - ref[:m, 0]).max())
        e_write = int(np.abs(cand[:m, 1] - ref[:m, 1]).max())
        worst_peak = max(worst_peak, abs(e_peak))
        worst_read = max(worst_read, e_read)
        worst_write = max(worst_write, e_write)
        rows.append((abs(e_peak), e_peak, peak_ref, peak_cand, len(ref), len(cand), c))
    print("%d configurations" % n)
    print("  frame idle cycles == solver total_gap, all carrying a read: %d exceptions" % idle_bad)
    print("  lead_in_block exact on (cycles, reads, writes): %d exceptions" % block_bad)
    print("  reads sum to n_read: %d exceptions" % short)
    print("  lead_in_node exact on (cycles, reads, writes): %d exceptions" % fold_bad)
    print("  worst lead-in size: %d entries folded, %d flat" % (entries, flat_entries))
    print("  composed peak occupancy vs the solver: worst error %d" % worst_peak)
    print("  composed cumulative reads / writes: worst %d / %d" % (worst_read, worst_write))
    for _, e, pr, pc, lr, lc, c in sorted(rows, key=lambda r: -r[0])[: args.top]:
        if e == 0:
            break
        print(
            "     peak %d -> %d (%+d)  period %d -> %d | k%s ifm%s s%s ch%d simd%d dw%d pw%d"
            % (
                pr,
                pc,
                e,
                lr,
                lc,
                c["k"],
                c["ifm_dim"],
                c["stride"],
                c["ifm_ch"],
                c["simd"],
                c["dw"],
                c["parallel_window"],
            )
        )
    return 1 if (idle_bad or block_bad or short or fold_bad or worst_peak) else 0


def _entries(node):
    """Run-length entries in a tree, which is what a leaf costs to carry."""
    if node.leaf:
        return len(node.sub_phases)
    return sum(_entries(k) for _, k in node.sub_phases)


def _budget_probe(p, windup, style):
    """``budget - n_first - (h-2)*n_mid`` as the shipped nest computes it.

    Read-only reproduction, to show *where* an exact windup breaks the read
    accounting. Nothing here is used by the constructor.
    """
    if style != "default":
        return None
    from finn.custom_op.fpgadataflow.convolutioninputgenerator import swg_nest_dims

    h, w, kh, kw, simd = swg_nest_dims(p)
    epw, beats = p["ELEM_PER_WINDOW"], kh * kw * simd
    if epw <= 0 or beats % epw or h < 1 or w < 1:
        return None
    steps = beats // epw
    per_row = min(
        (w - 1) * ((steps - 1) + p["TAIL_INCR_W"]) + (steps - 1) + p["TAIL_INCR_H"], w * beats
    )
    n_read, cap = p["LAST_READ_ELEM"] + 1, w * beats
    budget = max(0, n_read - windup)
    lead = max(0, p["BUF_ELEM_TOTAL"] - 1 - windup)
    n_first = min(budget if h == 1 else min(per_row + lead, budget), cap)
    return budget - n_first - max(0, h - 2) * min(per_row, cap)


def cmd_show(args):
    import json
    import swg_coupled as SC

    for c, inst, style, p in _iter(args.matrix):
        if args.cfg and json.dumps(c, sort_keys=True) != json.dumps(
            json.loads(args.cfg), sort_keys=True
        ):
            continue
        w = SC.swg_wait(p, style)
        share = swg_frame_reads(
            p, w["total_gap"], w["peak"], epw=p.get("ELEM_PER_WINDOW", 1), beats=w["beats"]
        )
        block = lead_in_block(share["cycles"], share["lead_in"], share["paced"])
        print(
            "k%s ifm%s s%s ch%d simd%d dw%d  windup %d  peak %d  lead_in %d reads %d writes"
            "  %d runs"
            % (
                c["k"],
                c["ifm_dim"],
                c["stride"],
                c["ifm_ch"],
                c["simd"],
                c["dw"],
                w["total_gap"],
                w["peak"],
                share["lead_in"],
                share["paced"],
                len(block),
            )
        )
        print("   %s%s" % (block[:6], " ..." if len(block) > 6 else ""))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--matrix", default="all")
    ck.add_argument("--budget", action="store_true", help="also probe the row read budget")
    ck.add_argument("--top", type=int, default=8)
    ck.set_defaults(fn=cmd_check)
    sh = sub.add_parser("show")
    sh.add_argument("--matrix", default="models")
    sh.add_argument("--cfg", default=None)
    sh.set_defaults(fn=cmd_show)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
