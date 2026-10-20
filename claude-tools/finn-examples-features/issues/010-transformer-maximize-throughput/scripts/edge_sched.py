"""Causal pass of a saved TAV model with the edge schedules logged: per edge the first-frame
write span and the consumer's read span. VARIANT as in size_offline.py.
usage: edge_sched.py <tav.onnx>"""
import inspect
import os
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.derive_characteristic as dc

sys.path.insert(0, os.path.dirname(__file__))
import size_offline  # noqa: E402,F401  (applies VARIANT, prints the depths)

orig = dc.edge_depth
rows = []


def logged(peak, read_times, edge, *a, **k):
    f = inspect.currentframe().f_back
    d = orig(peak, read_times, edge, *a, **k)
    w, r, n = np.sort(edge.writes), np.sort(read_times), edge.per_frame
    rows.append((f.f_locals["tensor"], f.f_locals["consumer"].name, w[0], w[n - 1], r[0],
                 r[min(n, len(r)) - 1], int(peak), int(d)))
    return d


dc.edge_depth = logged
m = ModelWrapper(sys.argv[1])
dc.derive_chained_tav_depths(m, int(m.get_metadata_prop("sizing_period")), causal=True)
for t, c, w0, w1, r0, r1, p, d in rows:
    print(f"{t[:26]:26s} -> {c[:22]:22s} w {w0:7d}..{w1:7d}  r {r0:7d}..{r1:7d}  peak {p:5d} depth {d:5d}")
