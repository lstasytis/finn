# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Prototype of the analytical SWG nest, iterated against the FSM oracle.

Workbench copy. The version that ships lives in
``src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py``.
"""

from finn.util.basic import Characteristic_Node


def nest_dims(p):
    """Trip counts of the five controller loops, outermost first (H..SIMD)."""
    inner = p["INNERMOST_STATE"]
    return [
        p["LOOP_%s_ITERATIONS" % s] + 2 + (1 if s == inner else 0)
        for s in ("H", "W", "KH", "KW", "SIMD")
    ]


def leaf(name, runs):
    return Characteristic_Node(name, [(int(n), list(v)) for n, v in runs if n > 0], True)


def comp(name, kids):
    return Characteristic_Node(name, [(int(n), c) for n, c in kids if n > 0 and c], False)


def clip(v, lo, hi):
    return max(lo, min(v, hi))


_c = clip


def default_tree(p):
    h, w, kh, kw, simd = nest_dims(p)
    epw = p["ELEM_PER_WINDOW"]
    beats = kh * kw * simd
    if epw <= 0 or beats % epw or h < 1 or w < 1:
        return None
    steps = beats // epw
    n_read = p["LAST_READ_ELEM"] + 1
    bms = p["TAIL_INCR_LAST"] + 1
    tw, th = p["TAIL_INCR_W"], p["TAIL_INCR_H"]

    def step(reads):
        return leaf("step", [(epw - reads, [0, 1]), (reads, [1, 1])])

    def window(name, reads):
        """``steps`` free-pointer steps taking ``reads`` words between them.

        The controller frees one slot per channel step and the window's whole
        draw on the last, so the last step takes the draw and the rest take one
        each -- and when the draw is bigger than a step can hold, the surplus
        spreads back over the earlier steps rather than being lost.
        """
        reads = clip(reads, 0, beats)
        if steps == 1:
            return step(reads)
        last = clip(reads - (steps - 1), 0, epw)
        q, r = divmod(reads - last, steps - 1)
        return comp(name, [(r, step(q + 1)), (steps - 1 - r, step(q)), (1, step(last))])

    def row(name, total):
        """One output row: ``w`` windows taking ``total`` words between them.

        A window releases its draw in one go, so a row whose draw does not fit
        in the beats it has left over stalls the fetch at the row boundary --
        the row runs `beats - epw` longer, which is the step the free pointer
        has not reached yet.
        """
        total = max(0, total)
        stretch = clip(total + beats - epw - w * beats, 0, beats - epw)
        inside = min(total, w * beats)
        ai = clip(a_int, 0, beats)
        if (w - 1) * ai + beats < inside:
            ai = clip(-(-(inside - beats) // (w - 1)), 0, beats)
        elif (w - 1) * ai > inside:
            ai = inside // (w - 1)
        ae = clip(inside - (w - 1) * ai, 0, beats)
        placed = (w - 1) * ai + ae
        spill = clip(total - placed, 0, stretch)
        node = comp(
            name,
            [
                (w - 1, window("win", ai)),
                (1, window("win_h", ae)),
                (1, leaf("stall", [(spill, [1, 0]), (stretch - spill, [0, 0])])),
            ],
        )
        return node, placed + spill

    cw_raw, ch_raw = (steps - 1) + tw, (steps - 1) + th
    if steps > 1 and (cw_raw > beats or ch_raw > beats):
        # a depthwise window whose draw is bigger than the window itself stalls
        # the fetch in a way the row-level stall does not describe
        return None
    cap = w * beats
    per_row = min((w - 1) * ((steps - 1) + tw) + (steps - 1) + th, cap)
    a_int = beats if per_row == cap else clip((steps - 1) + tw, 0, beats)

    # Wind-up: a whole window has to be in the buffer before the first beat can
    # be fetched. What that fill leaves over is still in hand when the beats
    # start, so the first row takes a longer draw than the others.
    windup = max(0, bms - beats + 2)
    lead = max(0, p["BUF_ELEM_TOTAL"] - 1 - windup)

    # A row that saturates its beats with reads is only ever one free-pointer
    # step behind. A row that does not saturate takes its row-end draw as a
    # burst, and whatever will not fit inside that window's beats stalls the
    # fetch until it has landed.
    stall = clip(per_row + beats - epw - cap, 0, beats - epw)
    if per_row < cap:
        stall += max(0, draw_h - beats)

    budget = n_read - windup
    first, n_first = row("row_first", min(per_row + lead, budget))
    mid, n_mid = row("row", per_row)
    tail_reads = n_read - windup - n_first - (h - 2) * n_mid
    last, n_last = row("row_last", tail_reads)
    # the frame reads its feature map exactly once: what the running draws have
    # not taken by the last beat drains at one word per cycle, which is what the
    # controller does once fetching is done
    drain = max(0, tail_reads - n_last)

    kids = [
        (1, first),
        (h - 2, mid),
        (1 if h > 1 else 0, last),
        (1, leaf("drain", [(drain, [1, 0])])),
        (1, leaf("fill", [(windup, [1, 0])])),
    ]
    return comp("swg_nest", kids)


def parallel_tree(p):
    h, w, kh, _, _ = nest_dims(p)
    if h < 1 or w < 2:
        return None
    n_read = p["LAST_READ_ELEM"] + 1
    gap_w = max(0, p["HEAD_INCR_W"] - 1)
    gap_h = max(0, p["HEAD_INCR_H"] - 1)
    fill = p["FIRST_WRITE_ELEM"] + 1
    row_len = (w - 1) * (kh + gap_w) + kh + gap_h
    # the parallel style holds one word per cycle for the whole frame: its
    # output is a whole window per beat, so the input stream is what paces it
    body = h * row_len
    fill = clip(n_read - body, 0, fill)
    pad = max(0, n_read - fill - body)

    def rows(reads_dense):
        v = [1, 0] if reads_dense else [0, 0]
        win = leaf("win", [(kh, [1, 1] if reads_dense else [0, 1]), (gap_w, v)])
        win_h = leaf("win_h", [(kh, [1, 1] if reads_dense else [0, 1]), (gap_h, v)])
        return comp("row", [(w - 1, win), (1, win_h)])

    dense = min(h, (n_read - fill) // row_len) if row_len else h
    return comp(
        "swg_nest",
        [
            (1, leaf("carry", [(1, [0, 0])])),
            (1, leaf("fill", [(fill, [1, 0])])),
            (dense, rows(True)),
            (h - dense, rows(False)),
            (1, leaf("pad", [(pad, [1, 0])])),
        ],
    )


def tree(p, style):
    return (default_tree if style == "default" else parallel_tree)(p)
