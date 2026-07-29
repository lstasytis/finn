"""Score every node-replay reference against its tree model, row 0 first.

``score_transformer_refs.py`` reports the worst error over *every* recorded
stream. That is the right thing for StreamingSplit and StreamingConcat, and
badly misleading for MVAU: its rtlsim reference has two input rows (the data
stream and the weight stream) and a tree model can only ever emit one, so the
weight row shows up as a six-figure "error" that no change to the schedule can
fix. The sizer reads row 0 and only row 0.

So this reports both, separately:

    r0 in / r0 out     the error on the schedule that actually sets FIFO depths
    xrow               the worst error on any *further* recorded row, i.e. the
                       cost of the single-row representation

    uv run python ci/experiments/score_nodes.py --op MVAU_hls -v
    uv run python ci/experiments/score_nodes.py --emit-budgets > /tmp/budgets.py
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tests.testing_util.tav_refs import (  # noqa: E402
    REF_DIR,
    as_rows,
    build_node,
    ensure_finn_env,
    read_tav,
    tree_tavs,
)


def score(entry):
    """None if no tree model (or no such op in this tree), else per-direction errors."""
    try:
        inst = build_node(entry["spec"])
        if inst.get_tree_model() is None:
            return None
    except Exception:
        # An op type, or an attribute value, that this FINN tree does not have.
        # The reference base is shared between two trees whose operator sets
        # have drifted (no ChannelwiseOp_hls here, no AccPool Function here).
        return "missing"
    tree_in, tree_out = tree_tavs(inst)
    out = {}
    for tree, key, tag in ((tree_in, "io_chrc_in", "in"), (tree_out, "io_chrc_out", "out")):
        t = as_rows(tree)[0].astype(np.int64)
        ref = as_rows(read_tav(entry[key])).astype(np.int64)
        n = min(t.size, ref.shape[-1])
        errs = [int(np.abs(t[:n] - ref[r, :n]).max()) if n else -1 for r in range(ref.shape[0])]
        out[tag] = dict(
            r0=errs[0],
            xrow=max(errs[1:]) if len(errs) > 1 else 0,
            nrows=len(errs),
            tree_len=int(t.size),
            ref_len=int(ref.shape[-1]),
            tree_tokens=int(t[-1]),
            ref_tokens=int(ref[0, -1]),
        )
    return out


def load(op_filter):
    refs = {}
    for path in sorted(glob.glob(os.path.join(REF_DIR, "*.json"))):
        for k, v in json.load(open(path)).items():
            if "spec" not in v:
                continue
            if op_filter and v["spec"]["op_type"] not in op_filter:
                continue
            refs[k] = v
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", action="append", default=[])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--worst", type=int, default=0, help="print the N worst per op type")
    ap.add_argument("--emit-budgets", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ensure_finn_env()
    refs = load(set(args.op))
    per_op = {}
    rows = {}
    for key in sorted(refs):
        got = score(refs[key])
        rows[key] = got
        op = refs[key]["spec"]["op_type"]
        a = per_op.setdefault(op, dict(n=0, modelled=0, exact=0, period=0, r0i=0, r0o=0, x=0,
                                       tok=0, missing=0, worst=[]))
        a["n"] += 1
        if got == "missing":
            a["missing"] += 1
            rows[key] = None
            continue
        if got is None:
            continue
        a["modelled"] += 1
        i, o = got["in"], got["out"]
        a["r0i"] = max(a["r0i"], i["r0"])
        a["r0o"] = max(a["r0o"], o["r0"])
        a["x"] = max(a["x"], i["xrow"], o["xrow"])
        a["exact"] += int(i["r0"] == 0 and o["r0"] == 0)
        a["period"] += int(i["tree_len"] == i["ref_len"])
        a["tok"] += int(i["tree_tokens"] == i["ref_tokens"] and o["tree_tokens"] == o["ref_tokens"])
        a["worst"].append((max(i["r0"], o["r0"]), key))
        if args.verbose:
            print(
                "%-30s r0 in %8d out %8d  xrow %8d  period %9d/%-9d  tok %8d/%-8d"
                % (key, i["r0"], o["r0"], max(i["xrow"], o["xrow"]),
                   i["tree_len"] // 2, i["ref_len"] // 2, i["tree_tokens"], i["ref_tokens"])
            )

    print("\n%-34s %5s %6s %8s %8s %8s %9s %9s"
          % ("op type", "refs", "modl", "exact", "r0 in", "r0 out", "period ok", "tokens ok"))
    print("-" * 92)
    for op in sorted(per_op, key=lambda o: -per_op[o]["r0i"] - per_op[o]["r0o"]):
        a = per_op[op]
        print("%-34s %5d %6d %8d %8d %8d %5d/%-5d %5d/%-5d%s"
              % (op, a["n"], a["modelled"], a["exact"], a["r0i"], a["r0o"],
                 a["period"], a["modelled"], a["tok"], a["modelled"],
                 "  (%d not in this tree)" % a["missing"] if a.get("missing") else ""))
        if args.worst:
            for e, k in sorted(a["worst"], reverse=True)[: args.worst]:
                r = rows[k]
                print("      %-28s r0 %8d  period %9d/%-9d  tok %8d/%-8d"
                      % (k, e, r["in"]["tree_len"] // 2, r["in"]["ref_len"] // 2,
                         r["in"]["tree_tokens"], r["in"]["ref_tokens"]))

    if args.emit_budgets:
        print("\n# --- generated by ci/experiments/score_nodes.py ---")
        for key in sorted(rows):
            g = rows[key]
            if g is None:
                continue
            print('    "%s": (%d, %d, %s),'
                  % (key, g["in"]["r0"], g["out"]["r0"],
                     g["in"]["tree_len"] == g["in"]["ref_len"]))
    if args.json_out:
        json.dump({k: v for k, v in rows.items() if v}, open(args.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
