"""Print the run-length structure of a stored reference next to its tree model.

A tree model *is* a run-length structure, so this shows directly which phase is
wrong: the wind-up, the burst length, the spacing, or the wrap.

    uv run python ci/experiments/node_rle.py MVAU_rtl_caa6a1
    uv run python ci/experiments/node_rle.py --op MVAU_rtl --list
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
    tree_tavs,
)


def load(op_filter=None):
    refs = {}
    for path in sorted(glob.glob(os.path.join(REF_DIR, "*.json"))):
        for k, v in json.load(open(path)).items():
            if "spec" not in v:
                continue
            if op_filter and v["spec"]["op_type"] != op_filter:
                continue
            refs[k] = v
    return refs


def rle(cum, n=None):
    """[(run length, tokens per cycle)] of a cumulative curve."""
    d = np.diff(np.concatenate(([0], np.asarray(cum, dtype=np.int64))))
    out, i = [], 0
    while i < len(d):
        j = i
        while j + 1 < len(d) and d[j + 1] == d[i]:
            j += 1
        out.append((int(j - i + 1), int(d[i])))
        i = j + 1
    return out if n is None else out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?")
    ap.add_argument("--op", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-n", type=int, default=16, help="phases to print")
    ap.add_argument("--attrs", default="", help="comma-separated attributes to print")
    args = ap.parse_args()

    from finn.util.basic import decompress_string_to_numpy

    ensure_finn_env()
    refs = load(args.op)
    if args.list:
        for k in sorted(refs):
            spec = refs[k]["spec"]
            keep = args.attrs.split(",") if args.attrs else None
            a = {n: v for n, v in spec["attrs"].items() if keep is None or n in keep}
            print("%-32s %s  %s" % (k, refs[k]["source"], a))
        return

    entry = refs[args.key]
    print("spec:", json.dumps(entry["spec"]["attrs"], sort_keys=True))
    inst = build_node(entry["spec"])
    print("folded in", inst.get_folded_input_shape(), "out", inst.get_folded_output_shape())
    tree = inst.get_tree_model()
    t_in, t_out = (None, None) if tree is None else tree_tavs(inst)
    for key, t in (("io_chrc_in", t_in), ("io_chrc_out", t_out)):
        ref = as_rows(decompress_string_to_numpy(entry[key])).astype(np.int64)
        print("\n== %s: ref %d rows x %d cycles (period %d), total %d"
              % (key, ref.shape[0], ref.shape[1], ref.shape[1] // 2, ref[0, -1]))
        for r in range(ref.shape[0]):
            print("   ref row%d: %s" % (r, rle(ref[r], args.n)))
        if t is not None:
            a = as_rows(t)[0].astype(np.int64)
            print("   tree    : %d cycles (period %d), total %d" % (a.size, a.size // 2, a[-1]))
            print("   tree rle: %s" % (rle(a, args.n),))
            n = min(a.size, ref.shape[1])
            diff = a[:n] - ref[0, :n]
            nz = np.flatnonzero(diff)
            print("   row0 err: max %d, first nonzero at cycle %s"
                  % (int(np.abs(diff).max()), int(nz[0]) if nz.size else None))


if __name__ == "__main__":
    main()
