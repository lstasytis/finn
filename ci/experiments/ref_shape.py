"""Extract the shape parameters of every stored reference schedule, as a table.

A tree model is a periodic burst schedule, so five numbers describe one stream:
where the first transaction lands, how long a burst is, how far apart bursts
are, how far apart transactions inside a burst are, and how long the period is.
Print those next to the folding, and a law is either visible or it is not.

    uv run python ci/experiments/ref_shape.py --op MVAU_rtl
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tests.testing_util.tav_refs import REF_DIR, as_rows, build_node, ensure_finn_env  # noqa: E402


def describe(cum):
    """(first, burst, gap, step, total) of a cumulative token curve."""
    d = np.diff(np.concatenate(([0], np.asarray(cum, dtype=np.int64))))
    hit = np.flatnonzero(d > 0)
    if hit.size == 0:
        return dict(first=None, burst=0, gap=0, step=0, total=0)
    steps = np.diff(hit)
    step = int(np.bincount(steps).argmax()) if steps.size else 0
    # a burst is a maximal run at the modal step
    brk = np.flatnonzero(steps != step) if steps.size else np.array([], dtype=int)
    burst = int(brk[0] + 1) if brk.size else int(hit.size)
    gap = int(steps[brk[0]]) if brk.size else 0
    return dict(
        first=int(hit[0]),
        burst=burst,
        gap=gap,
        step=step,
        total=int(np.asarray(cum)[-1]),
        n=int(hit.size),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True)
    ap.add_argument("--attrs", default="MW,MH,SIMD,PE,numInputVectors")
    args = ap.parse_args()

    from finn.util.basic import decompress_string_to_numpy

    ensure_finn_env()
    refs = {}
    for path in sorted(glob.glob(os.path.join(REF_DIR, "*.json"))):
        for k, v in json.load(open(path)).items():
            if "spec" in v and v["spec"]["op_type"] == args.op:
                refs[k] = v

    keep = args.attrs.split(",")
    print("%-24s %-40s %9s %9s | %8s %6s %7s %6s | %8s %6s %7s %6s"
          % ("key", "folding", "period", "ideal",
             "in first", "burst", "gap", "step", "out first", "burst", "gap", "step"))
    for k in sorted(refs):
        e = refs[k]
        a = e["spec"]["attrs"]
        inst = build_node(e["spec"])
        n_in = int(np.prod(inst.get_folded_input_shape()[:-1]))
        n_out = int(np.prod(inst.get_folded_output_shape()[:-1]))
        ri = as_rows(decompress_string_to_numpy(e["io_chrc_in"]))[0]
        ro = as_rows(decompress_string_to_numpy(e["io_chrc_out"]))[0]
        di, do = describe(ri), describe(ro)
        period = len(ri) // 2
        fold = " ".join("%s=%s" % (n, a.get(n)) for n in keep if n in a)
        print("%-24s %-40s %9d %9s | %8s %6d %7d %6d | %8s %6d %7d %6d"
              % (k, fold[:40], period, "%d/%d" % (n_in, n_out),
                 di["first"], di["burst"], di["gap"], di["step"],
                 do["first"], do["burst"], do["gap"], do["step"]))


if __name__ == "__main__":
    main()
