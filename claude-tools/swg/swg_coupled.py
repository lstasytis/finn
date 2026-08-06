# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The SWG's two waits, solved together instead of summed.

``swg_controller`` makes the generator wait twice in a frame, and the two waits
**compound**. A beat cannot leave until the input word it addresses has been
read; an input word cannot be read until the buffer slot it will overwrite has
been released, which happens when the write side reaches the next window. So the
read stream is not one word per cycle -- it is itself throttled -- and any law of
the form ``max_j (addr[j] - j)`` under-counts the demand wait by however far the
read stream is behind. No constant factor repairs that, because how far behind
the read stream is depends on the demand wait it is feeding.

Written out of ``swg_default_schedule``, with ``in0_V_V_TVALID`` and
``out_V_V_TREADY`` tied high, the FSM is four inequalities. Let ``fetch(j)`` be
the cycle beat ``j`` is fetched -- it leaves one cycle later, ``write_cmd`` being
registered -- and ``read(m)`` the cycle input word ``m`` is accepted::

    fetch(j) >= fetch(j-1) + 1                 one fetch per cycle
    fetch(j) >= read(addr[j]) + 1              fetch_cmd needs current <= newest
    read(m)  >= read(m-1) + 1                  one read per cycle
    read(m)  >= fetch(EPW * (W(m) - 1)) + 1    read_ok needs oldest < first_next

``addr[j]`` is the controller's address at beat ``j``. ``first_next`` steps by
``tail_incr`` on every ``EPW``-th fetch -- the start of a window -- so
``W(m) = min{w : T(w) > m - BUF_ELEM_TOTAL}`` is the window whose start releases
the slot word ``m`` needs, and ``EPW*(W(m)-1)`` is the beat that starts it.

Two things the FSM does that are not in those four lines:

* ``read_ok`` also wants ``oldest < current``. **It never binds on its own.**
  Instrumented over the whole configuration matrix, every stalled read has
  ``oldest < first_next`` false; the ``current`` term is never the sole reason.
  It is left out, and ``check`` would catch it if that ever stopped being true.
* A frame restarts only once *both* sides have finished it, and the read side
  stops dead at ``reading_done``. So where the write side finishes last, the
  first word of the next frame waits for the last beat of this one to leave:
  ``read(first of f+1) >= fetch(last of f) + 2``. That is the whole
  frame-boundary law; where the read side finishes last it is already implied by
  the read chain.

The parallel style is the same method with its own two gates: no line buffer, so
a read waits on ``newest <= current`` instead of on ``first_next``, and
``write_ok`` is the transaction rather than a registered ``write_cmd``. Both are
in ``gates()``; everything else is shared.

Both recurrences are monotone, so Kleene iteration from the schedule in which
neither side ever waits climbs to the least fixed point, which is the
earliest-possible and therefore the real schedule. Scored against
``swg_fsm``, four frames, every cycle of both streams, it is
**cycle-identical on all 384 configurations of the matrix** -- 225 default-style
and 159 parallel-style -- and ``delta()`` reproduces
``swg_fold.period_delta(aligned=True)`` array-for-array, in 40% of the time the
FSM takes.

    from swg_coupled import swg_wait, delta
    w = swg_wait(p, style)   # p, style from swg_fsm.swg_params / select_impl_style
    w["period"]              # the exact frame, from the coupled solve
    w["head"], w["row"]      # its idle cycles, by the two populations
    w["runs"]                # and their run lengths, as an FSM probe reports them
    delta(p, style)          # or the whole settled period, phased as the oracle is

    python claude-tools/swg/swg_coupled.py check --matrix all
    python claude-tools/swg/swg_coupled.py show --cfg '{"k":[3,3],...}'
"""

import argparse
import collections
import json
import numpy as np
import os
import sys

os.environ.setdefault(
    "FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swg_fsm import (  # noqa: E402
    SWG_LOOPS,
    SwgController,
    swg_default_schedule,
    swg_parallel_schedule,
    swg_params,
)

_MAX_PASSES = 4000


def frame_arrays(p):
    """Per beat of one frame: the address it emits and the level that stepped to it.

    Plus the ``tail_incr`` applied at each window start. Taken by running
    ``SwgController``, which is the same object the shipped model reads its
    parameters from, so the addresses here are the ones the hardware emits
    rather than a second derivation of them.
    """
    ctrl = SwgController(p)
    last_write, epw = p["LAST_WRITE_ELEM"], p["ELEM_PER_WINDOW"]
    addr, level, tails = [], [], []
    current, pos = 0, 0
    while True:
        addr.append(current)
        if pos == 0:
            tails.append(ctrl.tail_incr)
        step, state = ctrl.addr_incr, ctrl.state
        done = current == last_write
        ctrl.advance()
        level.append(state)
        pos = pos + 1 if pos != epw - 1 else 0
        if done:
            break
        current += step
    return (
        np.array(addr, dtype=np.int64),
        np.array(level, dtype=np.int64),
        np.array(tails, dtype=np.int64),
    )


def parallel_arrays(p):
    """The parallel style's addresses. It has no line buffer, so no tail chain."""
    ctrl = SwgController(p)
    last_write = p["LAST_WRITE_ELEM"]
    addr, current = [], p["FIRST_WRITE_ELEM"]
    while True:
        addr.append(current)
        done = current == last_write
        step = ctrl.addr_incr
        ctrl.advance()
        if done:
            break
        current += step
    return np.array(addr, dtype=np.int64)


def gates(p, style):
    """``(addr, gate, n_read, lag)``: what each beat waits for, and each word.

    ``gate[m]`` is the beat whose completion lets word ``m`` be read, or ``-1``
    where nothing holds it; ``lag`` is how many cycles after that beat the read
    may go. The two styles differ only here.

    *default*: the read waits on the line buffer -- ``oldest < first_next`` --
    and ``first_next`` steps by ``tail_incr`` at every ``EPW``-th fetch, so the
    gate is the beat that starts the releasing window.

    *parallel*: there is no line buffer. The read waits on ``newest <= current``,
    and ``current`` becomes ``addr[j]`` once beat ``j-1`` has gone, so the gate
    is the beat before the first one whose address reaches ``m-1``.
    """
    n_read = p["LAST_READ_ELEM"] + 1
    if style == "default":
        addr, _, tails = frame_arrays(p)
        epw, buf = p["ELEM_PER_WINDOW"], p["BUF_ELEM_TOTAL"]
        released = np.concatenate(([0], np.cumsum(tails)))
        window = np.searchsorted(released, np.arange(n_read) - buf, side="right")
        gate = np.where(window > 0, np.minimum((window - 1) * epw, addr.size - 1), -1)
        return addr, gate, n_read, 1
    addr = parallel_arrays(p)
    reach = np.maximum.accumulate(addr)
    first = np.searchsorted(reach, np.arange(n_read) - 1, side="left")
    gate = np.where(first < 1, -1, np.minimum(first, addr.size) - 1)
    return addr, gate, n_read, 1


def solve(p, frames=4, style="default"):
    """``(read, beat_cycle)`` in cycles over ``frames`` frames, or ``None``.

    Kleene iteration on the four inequalities, from the schedule in which
    neither side ever waits. Both are monotone, so it climbs to the least fixed
    point, which is the real schedule. For the default style ``beat_cycle`` is
    the *fetch*, and the transaction leaves one cycle later; for the parallel
    style ``write_ok`` is the transaction itself.
    """
    addr, gate, n_read, lag = gates(p, style)
    beats = addr.size
    if beats < 1 or n_read < 1:
        return None
    beat = np.arange(frames * beats, dtype=np.int64)
    word = np.arange(frames * n_read, dtype=np.int64)
    wants = np.tile(addr, frames) + np.repeat(np.arange(frames), beats) * n_read
    frees = np.tile(gate, frames) + np.repeat(np.arange(frames), n_read) * beats
    ungated = np.tile(gate, frames) < 0
    frees = np.clip(frees, 0, frames * beats - 1)
    # the read side stops dead at reading_done, so where the write side finishes
    # a frame last, the next frame's first word waits for its last beat to leave
    starts = np.arange(1, frames) * n_read
    ends = np.arange(1, frames) * beats - 1
    restart = 2 if style == "default" else 1

    read = word.copy()
    for _ in range(_MAX_PASSES):
        fetch = beat + np.maximum.accumulate(read[wants] + 1 - beat)
        want = np.where(ungated, -word, fetch[frees] + lag - word)
        want[starts] = np.maximum(want[starts], fetch[ends] + restart - starts)
        nxt = word + np.maximum.accumulate(want)
        if np.array_equal(nxt, read):
            return read, fetch
        read = nxt
    return None  # did not settle


def settled(p, frames=4, max_frames=16, style="default"):
    """One settled frame: ``(period, fetch, read, beats, n_read)``, or ``None``.

    The frame taken is the last one solved, and it is only accepted once it
    repeats its predecessor cycle for cycle -- a frame that still differs is
    still in the start-up transient, whatever its period says.
    """
    while frames <= max_frames:
        out = solve(p, frames, style)
        if out is None:
            return None
        read, fetch = out
        beats = fetch.size // frames
        n_read = read.size // frames
        for f in range(frames - 1, 1, -1):
            here = fetch[f * beats : (f + 1) * beats] - fetch[f * beats - 1]
            back = fetch[(f - 1) * beats : f * beats] - fetch[(f - 1) * beats - 1]
            if np.array_equal(here, back):
                period = int(fetch[(f + 1) * beats - 1] - fetch[f * beats - 1])
                return dict(
                    period=period,
                    fetch=fetch[f * beats : (f + 1) * beats],
                    before=int(fetch[f * beats - 1]),
                    read=read,
                    all=fetch,
                    frame=f,
                    beats=beats,
                    n_read=n_read,
                )
        frames *= 2
    return None


def swg_wait(p, style="default"):
    """Where a frame's cycles go, from the coupled solve. ``None`` if it never settles.

    ``period`` is the frame; ``beats`` of it are output transactions and the rest
    is the write stream standing still. Every one of those idle cycles is a beat
    waiting for the word it addresses, so they are reported by the loop level
    whose head increment carried the address out of reach -- which is the same
    decomposition an instrumented FSM produces, and the one a per-level law is
    trying to predict.

    The split into ``head`` and ``row`` is by **run signature**, because the two
    are not two independent waits and cannot be separated by which constraint
    binds. Every gap in this schedule is chain-bound -- the beat is waiting for a
    word that arrived as early as the throttled read stream allowed -- so asking
    "was this the free pointer or the demand?" returns "both, always". What *is*
    well defined is the length: a gap of exactly ``HEAD_INCR_x - 1`` is the
    address stepping out of reach by one head increment, and anything else is the
    reader having to catch up across a row. Those are the two populations the
    shipped model's two terms are trying to predict, and they are reported
    separately so the terms can be scored one at a time.
    """
    out = settled(p, style=style)
    if out is None:
        return None
    period, fetch, beats = out["period"], out["fetch"], out["beats"]
    n_read, f = out["n_read"], out["frame"]
    # Peak occupancy: how far the node's reads ever run ahead of its beats.
    # It only ever jumps on a read, so the maximum is attained at one, and it
    # falls out of the solved cycles without a period array. This is the term
    # the FIFO sizer keys on, and it is *not* the frame's idle-cycle count --
    # only part of the wait sits at the head of the frame.
    leaves = fetch + 1 if style == "default" else fetch
    when = out["read"][f * n_read : (f + 1) * n_read]
    ahead = np.arange(1, n_read + 1) - np.searchsorted(leaves, when, side="right")
    at = int(np.argmax(ahead))
    if style == "default":
        _, level, _ = frame_arrays(p)
    else:
        level = np.full(beats, -1, dtype=np.int64)
    gaps = np.diff(np.concatenate(([out["before"]], fetch))) - 1
    blame = np.concatenate(([level[-1]], level[:-1]))
    jumps = {abs(p["HEAD_INCR_%s" % name]) - 1 for name in SWG_LOOPS}
    head_mask = (gaps > 0) & np.isin(gaps, list(jumps))
    by_level = {}
    for lvl in list(range(len(SWG_LOOPS))) + [-1]:
        mask = (gaps > 0) & (blame == lvl)
        if mask.any():
            name = SWG_LOOPS[lvl] if lvl >= 0 else "FRAME"
            by_level[name] = (int(mask.sum()), int(gaps[mask].sum()))
    return dict(
        period=period,
        beats=int(beats),
        reads=int(n_read),
        peak=int(ahead[at]),
        peak_cycle=int(when[at] - (int(leaves[0]) - 1)),
        total_gap=int(period - beats),
        head=int(gaps[head_mask].sum()),
        row=int(gaps[(gaps > 0) & ~head_mask].sum()),
        by_level=by_level,
        runs=dict(collections.Counter(gaps[gaps > 0].tolist())),
    )


def delta(p, style="default"):
    """One settled period as a ``(cycles, 2)`` array of per-cycle read/write.

    Phased like ``swg_fold.period_delta(aligned=True)``: the window runs from the
    cycle after one frame wrap to the next wrap inclusive. A frame wraps once
    both sides have finished it, so the wrap is
    ``max(last read, last beat out)`` -- which is the same disjunction the FSM's
    two restart paths are.
    """
    out = settled(p, style=style)
    if out is None:
        return None
    read, fetch, beats, n_read = out["read"], out["all"], out["beats"], out["n_read"]
    # default: write_cmd is registered, so a beat leaves the cycle after its
    # fetch; parallel: write_ok is the transaction itself
    leaves = fetch + 1 if style == "default" else fetch
    f = out["frame"]
    if f < 1 or (f + 1) * beats > leaves.size or (f + 1) * n_read > read.size:
        return None
    wrap = lambda i: max(int(read[i * n_read - 1]), int(leaves[i * beats - 1]))  # noqa: E731
    lo, hi = wrap(f) + 1, wrap(f + 1)
    span = np.zeros((hi - lo + 1, 2), dtype=np.int64)
    span[leaves[(leaves >= lo) & (leaves <= hi)] - lo, 1] = 1
    span[read[(read >= lo) & (read <= hi)] - lo, 0] = 1
    return span


def _configs(matrix):
    from swg_configs import get_matrix

    return get_matrix(matrix)


def cmd_check(args):
    """Every default-style configuration, cycle for cycle against the FSM."""
    from swg_tav import build_swg

    exact = wrong = skipped = parallel = 0
    for c in _configs(args.matrix):
        try:
            _, inst = build_swg(c)
        except Exception:
            skipped += 1
            continue
        if "_rtl" not in type(inst).__name__:
            skipped += 1
            continue
        style = inst.select_impl_style()
        p = swg_params(inst, style)
        if p is None:
            skipped += 1
            continue
        out = solve(p, frames=args.frames, style=style)
        run = swg_default_schedule if style == "default" else swg_parallel_schedule
        sched, _ = run(p, n_feature_maps=args.frames)
        ref = np.array(sched, dtype=np.int8)
        if out is None:
            wrong += 1
            print("  NO FIXED POINT  %s" % json.dumps(c, sort_keys=True))
            continue
        read, fetch = out
        # default: write_cmd is registered, so a beat leaves the cycle after its
        # fetch. parallel: write_ok is the transaction.
        leaves = fetch + 1 if style == "default" else fetch
        span = int(leaves[-1]) + 2
        got = np.zeros((span, 2), dtype=np.int8)
        got[read[read < span], 0] = 1
        got[leaves[leaves < span], 1] = 1
        n = min(span, len(ref))
        parallel += style != "default"
        if np.array_equal(got[:n], ref[:n]):
            exact += 1
        else:
            wrong += 1
            i = int(np.flatnonzero(np.any(got[:n] != ref[:n], axis=1))[0])
            print("  DIVERGE at cycle %d of %d  %s" % (i, n, json.dumps(c, sort_keys=True)))
    print(
        "%d cycle-exact (%d of them parallel-style), %d wrong, %d skipped"
        % (exact, parallel, wrong, skipped)
    )
    return 1 if wrong else 0


def cmd_show(args):
    from swg_tav import build_swg

    for c in [json.loads(args.cfg)] if args.cfg else _configs(args.matrix):
        _, inst = build_swg(c)
        style = inst.select_impl_style()
        p = swg_params(inst, style)
        if p is None:
            continue
        w = swg_wait(p, style)
        if w is None:
            print("no settled frame: %s" % json.dumps(c, sort_keys=True))
            continue
        print(
            "k%s ifm%s s%s d%s ch%d simd%d dw%d"
            % (c["k"], c["ifm_dim"], c["stride"], c["dilation"], c["ifm_ch"], c["simd"], c["dw"])
        )
        print(
            "   period %d = beats %d + gap %d   (head jumps %d, row catch-up %d)"
            % (w["period"], w["beats"], w["total_gap"], w["head"], w["row"])
        )
        print("   by level %s" % w["by_level"])
        print("   gap runs %s" % w["runs"])
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--matrix", default="all")
    ck.add_argument("--frames", type=int, default=4)
    ck.set_defaults(fn=cmd_check)
    sh = sub.add_parser("show")
    sh.add_argument("--matrix", default="models")
    sh.add_argument("--cfg", default=None)
    sh.set_defaults(fn=cmd_show)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
