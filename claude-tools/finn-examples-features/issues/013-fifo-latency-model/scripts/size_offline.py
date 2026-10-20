"""Offline FIFO sizing of a saved TAV model (tav_model.py) with the derive_characteristic on
PYTHONPATH: the depths of DeriveFIFOSizes (max of the causal and non-causal pass), the total depth
and the total bits (depth x producer output stream width).
usage: size_offline.py <tav.onnx> [tensor ...]   (prints all edges with -v)"""
import inspect
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc

m = ModelWrapper(sys.argv[1])
period = int(m.get_metadata_prop("sizing_period"))
if "latency" in inspect.signature(dc.derive_chained_tav_depths).parameters:
    # fix/013: max over causal x latency
    p = [dc.derive_chained_tav_depths(m, period, c, la) for c in (0, 1) for la in (0, 1)]
    a = {t: max(p[0].get(t, 0), p[1].get(t, 0)) for t in set(p[0]) | set(p[1])}
    b = {t: max(p[2].get(t, 0), p[3].get(t, 0)) for t in set(p[2]) | set(p[3])}
else:
    a = dc.derive_chained_tav_depths(m, period, causal=False)
    b = dc.derive_chained_tav_depths(m, period, causal=True)
d = {t: max(a.get(t, 0), b.get(t, 0)) for t in set(a) | set(b)}
bits = 0
tot = 0
rows = []
for n in m.graph.node:
    for i, t in enumerate(n.output):
        if m.find_consumer(t) is None:
            continue
        dep = max(2, d.get(t, 0))
        try:
            w = getCustomOp(n).get_outstream_width(i)
        except Exception:
            w = 0
        tot += dep
        bits += dep * w
        c = m.find_consumer(t)
        rows.append((t, n.name, c.name, a.get(t, 0), b.get(t, 0), dep, w))
args = [x for x in sys.argv[2:] if x != "-v"]
for t, p, c, x, y, dep, w in rows:
    if "-v" in sys.argv or t in args:
        print(f"{t:28s} {p:34s} -> {c:34s} noncausal={x:7d} causal={y:7d} depth={dep:7d} w={w}")
print(f"TOTAL depth {tot} kbit {bits / 1024.0:.1f} period {period}")
