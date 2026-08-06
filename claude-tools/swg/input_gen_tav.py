# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fast harness for the ``input_gen`` tree model: no rtlsim, no vivado, no ipgen.

The reference is ``input_gen_ref.py``, the per-cycle transliteration of
``finn-rtllib/mvu_tiled/input_gen.sv``. ``dump`` freezes its token access
vectors for a configuration matrix into a golden file -- that is the slow part,
because it steps every cycle of every frame -- and ``check`` rebuilds them from
``input_gen_model.py`` and reports, per configuration, how far the model has
moved and in which direction.

Direction decides whether a model is safe. The FIFO sizer takes a depth from the
gap between a producer's cumulative writes and a consumer's cumulative reads, so:

* reading **earlier** than the reference shrinks the FIFO in front of the node;
* writing **later** than the reference shrinks the FIFO behind it.

Both are undersizing. The opposite two cost depth but never correctness. So
``undersize`` must stay at zero and ``oversize`` is the budget.

Usage
-----
    python claude-tools/swg/input_gen_tav.py dump  --matrix all
    python claude-tools/swg/input_gen_tav.py check --matrix all
    python claude-tools/swg/input_gen_tav.py check --live --matrix stress
    python claude-tools/swg/input_gen_tav.py show  --cfg '{"k":[3,3],...}'

``check`` exits non-zero when any configuration exceeds the thresholds or has
lost its model, so it works as the inner loop of an edit-test cycle and as a
pytest.
"""

import argparse
import json
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault(
    "FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from input_gen_model import nest_params, tree_model  # noqa: E402
from input_gen_ref import buf_size as ref_buf_size  # noqa: E402
from input_gen_ref import init_fp_inc as ref_fp_inc  # noqa: E402
from input_gen_ref import init_r_flag as ref_r_flag  # noqa: E402
from input_gen_ref import init_w as ref_w  # noqa: E402
from input_gen_ref import loop_nest_conv, tav  # noqa: E402
from swg_configs import get_matrix  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens/input_gen_ref.npz")


def cfg_key(c):
    return json.dumps(c, sort_keys=True)


def mvu_tiled_matrix():
    """The two nests ``mvu_tiled_axi.sv`` really instantiates ``input_gen`` with.

    Not sliding windows: the activation replay reads one group of ``TH`` vectors
    as a contiguous burst and replays it ``NF`` times, and the output reorder
    transposes ``(nf, th)`` into ``(th, nf)``. They are here because they are the
    module's only instantiations in the tree, so they are the only nests whose
    shape is not a proposal of this workbench's. Ranges are the ones
    ``mvau_tiled_tree`` is documented valid over.
    """
    out = []
    for th in (2, 3, 6, 9):
        for sf in (2, 4, 8, 16, 32, 64):
            for nf in (1, 2, 3, 6):
                out.append(dict(nest=[[nf, sf, th], [0, 1, sf], sf * th], tag="mvu replay"))
                out.append(dict(nest=[[th, nf], [1, th], nf * th], tag="mvu reorder"))
    return out


def matrix(name):
    return mvu_tiled_matrix() if name == "mvu" else get_matrix(name)


def nest_of(c):
    """The loop nest this configuration instantiates ``input_gen`` with, or ``None``.

    ``parallel_window`` emits a whole ``k*k`` window per beat, so its output word
    is wider than its input word. ``input_gen`` has one ``DATA_WIDTH`` for both,
    so it cannot express that folding at all -- the configuration is out of the
    module's scope rather than out of the model's.
    """
    if "nest" in c:
        return tuple(c["nest"])
    if c["parallel_window"]:
        return None
    return loop_nest_conv(
        c["ifm_dim"], c["k"], c["stride"], c["dilation"], c["ifm_ch"], c["simd"], c["dw"]
    )


def reference(c):
    """(period, cumulative reads, cumulative writes) from the per-cycle model."""
    nest = nest_of(c)
    if nest is None:
        return None
    settled = tav(*nest)
    if settled is None:
        return None
    period, rd, wr = settled
    return dict(inp=rd.astype(np.int64), out=wr.astype(np.int64), period=int(period))


def candidate(c):
    """The same, from the tree model."""
    nest = nest_of(c)
    if nest is None:
        return None
    tree = tree_model(*nest)
    if tree is None:
        return None
    cum = tree.cumulative(periods=1)
    return dict(
        inp=cum[:, 0].astype(np.int64),
        out=cum[:, 1].astype(np.int64),
        period=int(cum.shape[0]),
        tree_name=tree.name,
        leaves=count_leaves(tree),
    )


def count_leaves(node):
    if node.leaf:
        return len(node.sub_phases)
    return sum(count_leaves(sub) for _, sub in node.sub_phases)


def compare(gold, new):
    """Signed divergence of a candidate TAV against the reference one."""
    n = min(len(gold["inp"]), len(new["inp"]))
    m = min(len(gold["out"]), len(new["out"]))
    din = new["inp"][:n] - gold["inp"][:n]
    dout = new["out"][:m] - gold["out"][:m]
    tokens = max(int(gold["inp"][-1]), int(gold["out"][-1]), 1)
    undersize = max(int(din.max(initial=0)), int((-dout).max(initial=0)))
    oversize = max(int((-din).max(initial=0)), int(dout.max(initial=0)))
    return dict(
        undersize=undersize,
        oversize=oversize,
        undersize_frac=undersize / tokens,
        oversize_frac=oversize / tokens,
        period_gold=int(gold["period"]),
        period_new=int(new["period"]),
        period_frac=abs(int(new["period"]) - int(gold["period"])) / max(int(gold["period"]), 1),
        tokens=tokens,
    )


def cmd_dump(args):
    cfgs = matrix(args.matrix)
    store, skipped, t0 = {}, [], time.time()
    for c in cfgs:
        try:
            r = reference(c)
        except Exception as e:
            skipped.append((cfg_key(c), "%s: %s" % (type(e).__name__, e)))
            continue
        if r is None:
            skipped.append((cfg_key(c), "no reference (parallel_window, or never settles)"))
            continue
        store[cfg_key(c)] = np.array(r, dtype=object)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **store)
    print(
        "dumped %d/%d configs to %s in %.1fs (%d skipped)"
        % (len(store), len(cfgs), args.out, time.time() - t0, len(skipped))
    )
    if args.verbose:
        for key, why in skipped:
            print("  skip %s -- %s" % (key, why))


def cmd_check(args):
    if args.live:
        gold = None
        cfgs = matrix(args.matrix or "all")
    else:
        z = np.load(args.golden, allow_pickle=True)
        gold = {k: z[k].item() for k in z.files}
        cfgs = matrix(args.matrix) if args.matrix else [json.loads(k) for k in gold]
    rows, missing, out_of_scope, t0 = [], [], 0, time.time()
    leaves = 0
    for c in cfgs:
        key = cfg_key(c)
        if nest_of(c) is None:
            out_of_scope += 1
            continue
        ref = reference(c) if gold is None else gold.get(key)
        if ref is None:
            out_of_scope += 1
            continue
        try:
            new = candidate(c)
        except Exception as e:
            rows.append((c, dict(error="%s: %s" % (type(e).__name__, e))))
            continue
        if new is None:
            missing.append(key)
            continue
        leaves = max(leaves, new["leaves"])
        rows.append((c, compare(ref, new)))
    bad = []
    for c, r in rows:
        if "error" in r:
            bad.append((c, r))
        elif (
            (r["undersize"] > args.const and r["undersize_frac"] > args.fail_under)
            or (r["oversize"] > args.const and r["oversize_frac"] > args.fail_over)
            or r["period_frac"] > args.fail_period
        ):
            bad.append((c, r))
    print(
        "checked %d configs in %.2fs (%d out of the module's scope, worst tree %d leaves)"
        % (len(rows), time.time() - t0, out_of_scope, leaves)
    )
    scored = [r for _, r in rows if "error" not in r]
    if scored:
        print(
            "worst undersize %.4f  oversize %.4f  period %.4f"
            % (
                max(r["undersize_frac"] for r in scored),
                max(r["oversize_frac"] for r in scored),
                max(r["period_frac"] for r in scored),
            )
        )
    if missing:
        print("%d configs declined by the model (would fall back to rtlsim):" % len(missing))
        for key in missing[: args.top]:
            print("  " + key)
    for c, r in bad[: args.top]:
        print("FAIL %s\n     %s" % (cfg_key(c), r))
    print("%d/%d configs outside budget" % (len(bad), len(rows)))
    return 1 if (bad or (missing and not args.allow_missing)) else 0


def cmd_show(args):
    c = json.loads(args.cfg)
    nest = nest_of(c)
    if nest is None:
        print("outside input_gen's scope (parallel_window)")
        return 1
    dims, coefs, fm = nest
    ref, new = reference(c), candidate(c)
    print("dims=%s coefs=%s fm=%d" % (dims, coefs, fm))
    if new is None:
        print("no tree model for this config")
        return 1
    print(
        "model period=%d in=%d out=%d leaves=%d"
        % (new["period"], new["inp"][-1], new["out"][-1], new["leaves"])
    )
    if ref is not None:
        print("ref   period=%d in=%d out=%d" % (ref["period"], ref["inp"][-1], ref["out"][-1]))
        print(compare(ref, new))
    return 0


def cmd_elab(args):
    """The model re-derives the module's elaboration; check it against the reference's.

    Both files transliterate ``INIT_W`` / ``INIT_R_FLAG`` / ``INIT_FP_INC`` /
    ``INIT_MAX_OCCUPANCY`` from the SV independently. The dynamics are checked
    against each other by ``check``; a slip in the elaboration would agree with
    itself and pass, so it is checked here instead.
    """
    bad = 0
    nests = [nest_of(c) for c in matrix("all") + matrix("mvu")]
    for nest in [n for n in nests if n is not None]:
        dims, coefs, fm = nest
        w, r_flag, fp_inc, buf = nest_params(dims, coefs, fm)
        w_r = ref_w(dims, coefs, fm)
        r_flag_r = ref_r_flag(dims, coefs, w_r)
        fp_inc_r = ref_fp_inc(dims, coefs, w_r, r_flag_r)
        if (w, r_flag, fp_inc, buf) != (w_r, r_flag_r, fp_inc_r, ref_buf_size(dims, coefs, fm)):
            bad += 1
            if bad <= args.top:
                print("ELAB MISMATCH dims=%s coefs=%s fm=%d" % (dims, coefs, fm))
    print("%d/%d elaborations disagree with the reference" % (bad, len(nests)))
    return 1 if bad else 0


def cmd_drift(args):
    """Does the model's error stay constant across frames, or grow per frame?

    A constant offset is a whole-node latency and the sizer absorbs it. An error
    that grows every period is a token the node does not deliver, and the
    occupancy sum the sizer integrates drifts by that much every frame -- which
    is how a post-hoc wind-up shift wrecks a tree that already models its own.
    So the test is not "how big is the error" but "is it the same at period 1 and
    at period 4".
    """
    worst, checked = [], 0
    for c in matrix(args.matrix or "all"):
        if nest_of(c) is None:
            continue
        ref, new = reference(c), candidate(c)
        if ref is None or new is None:
            continue
        checked += 1
        period, reads, writes = ref["period"], ref["inp"], ref["out"]
        tree = tree_model(*nest_of(c))
        cum = tree.cumulative(periods=args.periods)
        if cum.shape[0] != args.periods * period:
            worst.append((abs(cum.shape[0] - args.periods * period), "period", cfg_key(c)))
            continue
        long_ref = np.concatenate(
            [reads + i * int(reads[-1]) for i in range(args.periods)]
        ), np.concatenate([writes + i * int(writes[-1]) for i in range(args.periods)])
        for row, name in ((0, "in"), (1, "out")):
            err = cum[:, row] - long_ref[row]
            per_frame = [
                int(err[(i + 1) * period - 1]) - int(err[i * period - 1] if i else 0)
                for i in range(args.periods)
            ]
            if len(set(per_frame[1:])) > 1 or (per_frame and per_frame[-1] != 0):
                worst.append((max(abs(x) for x in per_frame), name, cfg_key(c)))
    print("%d configs, %d with a per-frame drift" % (checked, len(worst)))
    for row in sorted(worst, reverse=True)[: args.top]:
        print("  drift %d on %s: %s" % row)
    return 1 if worst else 0


def cmd_fuzz(args):
    """Random loop nests, model against the per-cycle reference.

    The matrix is convolutions; this is not. It exists to reach nests the
    sliding-window shapes never produce -- zero coefficients, transposed levels,
    coefficients that overlap, buffers that hold a hundred frames -- and to make
    sure the model either matches the reference on them or declines.
    """
    rng = np.random.default_rng(args.seed)
    checked = declined = bad = 0
    for _ in range(args.n):
        d = int(rng.integers(1, 5))
        dims = [int(rng.integers(1, 5)) for _ in range(d)]
        coefs = [int(rng.integers(0, 7)) for _ in range(d)]
        span = sum((n - 1) * c for n, c in zip(dims, coefs))
        # a quarter of them are given a feature map too small for the nest, to
        # exercise the declines rather than only the well-formed path
        fm = span + 1 + int(rng.integers(0, 4))
        if rng.random() < 0.25:
            fm = int(rng.integers(1, span + 2))
        if int(np.prod(dims)) * fm > 20000:
            continue
        cand = tree_model(dims, coefs, fm)
        if cand is None:
            declined += 1
            continue
        settled = tav(dims, coefs, fm)
        if settled is None:
            continue
        checked += 1
        period, rd, wr = settled
        cum = cand.cumulative(periods=1)
        if (
            cum.shape[0] != period
            or not np.array_equal(cum[:, 0], rd)
            or not np.array_equal(cum[:, 1], wr)
        ):
            bad += 1
            if bad <= args.top:
                print(
                    "FUZZ MISMATCH dims=%s coefs=%s fm=%d period ref=%d model=%d"
                    % (dims, coefs, fm, period, cum.shape[0])
                )
    print("%d nests matched exactly, %d declined, %d wrong" % (checked - bad, declined, bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--matrix", default="all", choices=["pytest", "models", "stress", "mvu", "all"])
    d.add_argument("-o", "--out", default=GOLDEN)
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(fn=cmd_dump)

    c = sub.add_parser("check")
    c.add_argument("-g", "--golden", default=GOLDEN)
    c.add_argument("--live", action="store_true", help="rerun the per-cycle reference instead")
    c.add_argument("--matrix", default=None, choices=["pytest", "models", "stress", "mvu", "all"])
    c.add_argument("--fail-under", type=float, default=0.0)
    c.add_argument("--const", type=int, default=0)
    c.add_argument("--fail-over", type=float, default=0.10)
    c.add_argument("--fail-period", type=float, default=0.05)
    c.add_argument("--allow-missing", action="store_true")
    c.add_argument("--top", type=int, default=20)
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser("show")
    s.add_argument("--cfg", required=True)
    s.set_defaults(fn=cmd_show)

    dr = sub.add_parser("drift")
    dr.add_argument("--matrix", default=None, choices=["pytest", "models", "stress", "mvu", "all"])
    dr.add_argument("--periods", type=int, default=4)
    dr.add_argument("--top", type=int, default=10)
    dr.set_defaults(fn=cmd_drift)

    e = sub.add_parser("elab")
    e.add_argument("--top", type=int, default=10)
    e.set_defaults(fn=cmd_elab)

    f = sub.add_parser("fuzz")
    f.add_argument("-n", type=int, default=2000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--top", type=int, default=10)
    f.set_defaults(fn=cmd_fuzz)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
