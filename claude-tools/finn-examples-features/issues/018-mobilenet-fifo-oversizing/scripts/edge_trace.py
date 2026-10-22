"""Issue 018: dump one edge's chained write/read schedule (one pass) and where its peak is.
usage: edge_trace.py <tav.onnx> <producer> <consumer> [causal latency]"""
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.derive_characteristic as dc

m = ModelWrapper(sys.argv[1])
prod, cons = sys.argv[2], sys.argv[3]
c, la = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 0)
gp = int(m.get_metadata_prop("sizing_period"))
orig = dc.peak_occupancy
got = {}


def wrapped(w, r, period, per_frame, fill_only):
    f = sys._getframe(1).f_locals
    node = f.get("node")
    t = f.get("t")
    if node is not None and node.name == cons and m.find_producer(t).name == prod:
        got.update(w=np.asarray(w), r=np.asarray(r), pf=per_frame)
    return orig(w, r, period, per_frame, fill_only)


dc.peak_occupancy = wrapped
dc.derive_chained_tav_depths(m, gp, c, la)
w, r, pf = np.sort(got["w"]), np.sort(got["r"]), got["pf"]
print("gp", gp, "per_frame", pf, "writes", len(w), "reads", len(r))
for f in range(2):
    ws, rs = w[f * pf:(f + 1) * pf], r[f * pf:(f + 1) * pf]
    print(f"frame {f}: writes {ws[0]}..{ws[-1]} ({(ws[-1]-ws[0])/gp:.3f} gp), "
          f"reads {rs[0]}..{rs[-1]} ({(rs[-1]-rs[0])/gp:.3f} gp)")
    ev = np.unique(np.concatenate([ws, rs]))
    occ = np.searchsorted(ws, ev, side="right") - np.searchsorted(rs, ev, side="right")
    k = int(np.argmax(occ))
    print(f"  peak {occ[k]} at t={ev[k]} (written {np.searchsorted(ws, ev[k], side='right')}, "
          f"read {np.searchsorted(rs, ev[k], side='right')})")
    # occupancy profile at 20 points
    pts = np.linspace(min(ws[0], rs[0]), max(ws[-1], rs[-1]), 21).astype(int)
    o = np.searchsorted(ws, pts, side="right") - np.searchsorted(rs, pts, side="right")
    print("  profile", list(o))
