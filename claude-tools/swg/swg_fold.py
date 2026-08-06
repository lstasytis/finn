# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Workbench: fold an SWG period into the controller's loop nest.

Uses the committed FSM execution as an *oracle* -- it produces the per-cycle
schedule, this folds it by the five loop levels and reports what structure comes
out. The point is to read the closed form off the fold: what a level's body
looks like, whether every iteration of a level is identical, and where the
exceptions are.

    python claude-tools/swg/swg_fold.py --matrix models
    python claude-tools/swg/swg_fold.py --cfg '{"k":[3,3],"ifm_dim":[113,113],...}' -v
"""

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(__file__))

from swg_configs import get_matrix  # noqa: E402
from swg_tav import build_swg  # noqa: E402

from swg_fsm import (  # noqa: E402
    SWG_LOOPS,
    swg_default_schedule,
    swg_parallel_schedule,
    swg_params,
)

LEVEL_NAMES = ("H", "W", "KH", "KW", "SIMD")  # outermost first


def nest_dims(p):
    """True trip counts of the five controller loops, outermost first.

    ``LOOP_x_ITERATIONS`` is the counter's reload value, which the code
    generator sets to ``trips - 2``; the innermost level loses one more because
    the FSM starts already in its state.
    """
    inner = p["INNERMOST_STATE"]
    dims = []
    for name in LEVEL_NAMES:
        t = p["LOOP_%s_ITERATIONS" % name] + 2
        if name == inner:
            t += 1
        dims.append(t)
    return dims


def period_delta(inst, aligned=False):
    """One steady-state period of ``inst`` as a (cycles, 2) array, plus its params.

    ``aligned`` takes the frame that starts at a restart, which is the phase the
    loop nest is written in; the default takes the phase the committed model
    emits (one cycle before a period boundary), which is what rtlsim records.
    """
    impl_style = inst.select_impl_style()
    p = swg_params(inst, impl_style)
    if p is None:
        return None, None
    run = swg_default_schedule if impl_style == "default" else swg_parallel_schedule
    sched, restarts = run(p, n_feature_maps=4)
    if len(restarts) < 4:
        return None, p
    period = restarts[3] - restarts[2]
    start = restarts[2] + 1 if aligned else 2 * period - 1
    if start + period > len(sched):
        return None, p
    return np.array(sched[start : start + period], dtype=np.int64), p


def runs_of(delta):
    if delta.shape[0] == 0:
        return []
    cut = np.flatnonzero(np.any(np.diff(delta, axis=0) != 0, axis=1)) + 1
    start = np.concatenate(([0], cut))
    length = np.diff(np.concatenate((start, [delta.shape[0]])))
    return [(int(a), (int(v[0]), int(v[1]))) for a, v in zip(length, delta[start])]


def split_by_writes(delta, n):
    """``n`` equal-write blocks, or None if the period does not cut that way."""
    total = np.cumsum(delta[:, 1])
    if total[-1] == 0 or total[-1] % n:
        return None
    share = total[-1] // n
    end = np.searchsorted(total, share * np.arange(1, n + 1), side="left") + 1
    if end[-1] != delta.shape[0]:
        return None
    return [delta[(0 if j == 0 else end[j - 1]) : end[j]] for j in range(n)]


def rle_blocks(blocks):
    """Run-length the block sequence: [(count, block), ...] over consecutive equals."""
    out = []
    for b in blocks:
        if out and out[-1][1].shape == b.shape and np.array_equal(out[-1][1], b):
            out[-1][0] += 1
        else:
            out.append([1, b])
    return out


def describe(delta, dims, level=0, path="", out=None, verbose=False):
    """Recursive fold; collects a text description of the structure."""
    if out is None:
        out = []
    pad = "  " * level
    if level >= len(dims):
        out.append(f"{pad}{path or 'leaf'}: LEAF {len(runs_of(delta))} runs, {len(delta)} cycles")
        if verbose:
            out.append(f"{pad}   {runs_of(delta)[:24]}")
        return out
    n = dims[level]
    name = LEVEL_NAMES[level]
    if n < 2:
        return describe(delta, dims, level + 1, path, out, verbose)
    blocks = split_by_writes(delta, n)
    if blocks is None:
        out.append(f"{pad}{path}{name}x{n}: NO CUT -> leaf {len(runs_of(delta))} runs")
        if verbose:
            out.append(f"{pad}   {runs_of(delta)[:24]}")
        return out
    groups = rle_blocks(blocks)
    shape = "+".join(f"{cnt}x{chr(ord('a') + i)}" for i, (cnt, _) in enumerate(groups))
    out.append(f"{pad}{path}{name}x{n}: {len(groups)} groups  [{shape}]")
    for i, (cnt, u) in enumerate(groups):
        describe(u, dims, level + 1, f"{path}{name}[{chr(ord('a') + i)}x{cnt}]/", out, verbose)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    cfgs = [json.loads(a.cfg)] if a.cfg else get_matrix(a.matrix or "models")[: a.n]
    for c in cfgs:
        model, inst = build_swg(c)
        delta, p = period_delta(inst)
        print("=" * 78)
        print(json.dumps(c, sort_keys=True))
        if delta is None:
            print("  no period")
            continue
        dims = nest_dims(p)
        print(
            f"  style={inst.select_impl_style()} inner={p['INNERMOST_STATE']} "
            f"dims(H,W,KH,KW,SIMD)={dims} period={len(delta)} "
            f"reads={int(delta[:,0].sum())} writes={int(delta[:,1].sum())} "
            f"flat_runs={len(runs_of(delta))}"
        )
        for line in describe(delta, dims, verbose=a.v):
            print("  " + line)


if __name__ == "__main__":
    main()
