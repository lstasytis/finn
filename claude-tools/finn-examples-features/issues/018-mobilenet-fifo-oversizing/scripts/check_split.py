"""Issue 018: does peak_occupancy evaluate the two frames separately (split) on an edge, or
jointly (both repeated every period, per-frame peak counted twice)? Causal+latency pass.
usage: check_split.py <tav.onnx>"""
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.derive_characteristic as dc

m = ModelWrapper(sys.argv[1])
orig = dc.peak_occupancy
rec = {}


def wrapped(w, r, period, per_frame, fill_only):
    t = sys._getframe(1).f_locals.get("t")
    n = min(len(w), len(r))
    split = bool(per_frame and 2 * per_frame <= n)
    p = orig(w, r, period, per_frame, fill_only)
    joint = dc.periodic_peak(np.sort(w[:n]), np.sort(r[:n]), period) if n else 0
    rec[t] = (split, n, per_frame, p, joint)
    return p


dc.peak_occupancy = wrapped
dc.derive_chained_tav_depths(m, int(m.get_metadata_prop("sizing_period")), True, True)
ns = sum(1 for v in rec.values() if not v[0])
print(f"edges {len(rec)}, joint (unsplit) {ns}")
for t, v in sorted(rec.items(), key=lambda x: -x[1][3])[:20]:
    print(f"{t:30s} split={v[0]} n={v[1]} per_frame={v[2]} peak={v[3]} joint={v[4]}")
