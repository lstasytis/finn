"""Diff the chained_tav per-edge derivation between two TAV sources.

`tree_rank.py` says *which* edges a tree model moves. This says *why*: it runs
the sizing pass twice with `DeriveFIFOSizes.chained_tav_trace` collected and
prints, side by side, every term the pass computed for the edges that changed --
peak, per_frame, t_up, allowance, after_slack, floor, down_chain.

    uv run python ci/experiments/tree_trace_diff.py cnv-w2a2 --tree only:MVAU_hls
"""

import argparse
import contextlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tests.testing_util.tav_refs import ensure_finn_env  # noqa: E402

FIELDS = [
    "peak", "per_frame", "t_up", "allowance", "after_slack", "floor", "capped",
    "run", "burst", "cons_period", "down_chain", "drives_pacer", "burst_floor",
    "join", "depth",
]


def collect(cfg, tree, strategy="chained_tav"):
    """Size once with the trace on. Returns {(tensor, consumer): row}."""
    import tr_loop
    from finn.transformation.fpgadataflow import derive_characteristic as dc

    # ``DeriveFIFOSizes`` only threads its trace into the phased pass; the final
    # depth is the max over the phased and the causal pass, so an edge whose
    # depth is set by the causal pass never appears. Trace both and tag them.
    rows = []
    orig = dc.derive_chained_tav_depths

    def patched(model, *a, **kw):
        kw["trace"] = rows
        causal = kw.get("causal", False)
        n0 = len(rows)
        out = orig(model, *a, **kw)
        # DWC output tensors are named randomly per run, so a tensor name is not
        # a stable key across two sizing runs. Re-key on (producer, port).
        who = {}
        for n in model.graph.node:
            for i, t in enumerate(n.output):
                who[t] = "%s#%d" % (n.name, i)
        for r in rows[n0:]:
            r["causal"] = causal
            r["edge"] = who.get(r.get("tensor"), r.get("tensor"))
        return out

    dc.derive_chained_tav_depths = patched
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            tr_loop.size_model(cfg, strategy, tree)
    finally:
        dc.derive_chained_tav_depths = orig
    out = {}
    for r in rows:
        if "tensor" not in r:
            continue
        out[(r["edge"], r.get("consumer"), r.get("causal"))] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tree", default="prefer")
    ap.add_argument("--ref", default="none")
    ap.add_argument("--strategy", default="chained_tav")
    ap.add_argument("--grep", default="")
    args = ap.parse_args()

    ensure_finn_env()
    from tr_loop import MODELS

    cfg = MODELS[args.model]
    a = collect(cfg, args.ref, args.strategy)
    b = collect(cfg, args.tree, args.strategy)
    n = 0
    for k in sorted(set(a) & set(b), key=str):
        ra, rb = a[k], b[k]
        if ra.get("depth") == rb.get("depth"):
            continue
        if args.grep and args.grep not in str(k):
            continue
        n += 1
        print("=" * 78)
        print("%s -> %s [causal=%s] : depth %s -> %s" % (k[0], k[1], k[2], ra.get("depth"), rb.get("depth")))
        for f in FIELDS:
            if f in ra or f in rb:
                va, vb = ra.get(f), rb.get(f)
                mark = "   <<<" if va != vb else ""
                print("   %-14s %14s  %14s%s" % (f, va, vb, mark))
    print("\n%d edges changed depth" % n)


if __name__ == "__main__":
    main()
