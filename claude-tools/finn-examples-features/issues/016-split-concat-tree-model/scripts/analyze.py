"""Summarize measure.py results: schedule offsets/interval and tree-vs-rtlsim TAV deltas."""
import glob
import io
import json
import sys
from contextlib import redirect_stdout

import numpy as np

from finn.util.test import compare_chr_funcs_up_to_rotation, compare_two_chr_funcs


def stored_window(trace, cycles, P=5):
    t = [int(c) for c in trace]
    period = cycles // P
    wrap = period * P
    t += [0] * max(0, wrap - len(t))
    start = max(0, P - 3) * period
    return np.cumsum([t[(start + i) % wrap] for i in range(2 * period)]), period


files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "m_*.json"))
for f in files:
    r = json.load(open(f))
    tr = r["traces"]
    first = {k: v.find("1") for k, v in tr.items()}
    # interval: median gap between transfers on the "wide" side
    wide = "in0" if r["op"] == "split" else "out0"
    idx = [i for i, c in enumerate(tr[wide]) if c == "1"]
    gaps = np.diff(idx)
    ii = int(np.median(gaps)) if len(gaps) else -1
    # per narrow stream: offset of first transfer from the first wide transfer, in units of ii
    rin, period = stored_window(tr["in0"], r["cycles"])
    rout, _ = stored_window(tr["out0"], r["cycles"])
    tin = np.array(r["tree_in"])
    tout = np.array(r["tree_out"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok_in = compare_two_chr_funcs(tin, rin, 0.0, 0.0, 0, 0)
        ok_out = compare_two_chr_funcs(tout, rout, 0.0, 0.0, 0, 0)
        rot = compare_chr_funcs_up_to_rotation(tin, tout, rin, rout, 0.0, 1.0, 0, 10**6)
    din = int(np.max(np.abs(tin[: min(len(tin), len(rin))] - rin[: min(len(tin), len(rin))])))
    dout = int(np.max(np.abs(tout[: min(len(tout), len(rout))] - rout[: min(len(tout), len(rout))])))
    print(
        "%-6s ch=%-12s simd=%d nvec=%-5s prag=%d | cyc=%4d per(rtl)=%4d per(tree)=%4d ii=%d first=%s"
        " | exact-peak-delta in/out %d/%d, rot (rot,in,out)=%s"
        % (
            r["op"],
            ",".join(map(str, r["chans"])),
            r["simd"],
            ",".join(map(str, r["nvec"])),
            r["pragma"],
            r["cycles"],
            period,
            len(tin) // 2,
            ii,
            first,
            din,
            dout,
            rot[1:],
        )
    )
