"""Issue 015: offline FIFO sizing of a saved TAV model (013's tav_model.py) with the
derive_characteristic on PYTHONPATH, as DeriveFIFOSizes does it: max over causal x latency,
then the join floors (join_window_floors) where the sizer has them.
usage: size_offline.py <tav.onnx> [-v]   (-v: all edges; join floors are always printed)"""
import inspect
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
period = int(m.get_metadata_prop("sizing_period"))
if "latency" in inspect.signature(dc.derive_chained_tav_depths).parameters:
    passes = [dc.derive_chained_tav_depths(m, period, c, la) for c in (0, 1) for la in (0, 1)]
else:
    passes = [dc.derive_chained_tav_depths(m, period, causal=c) for c in (0, 1)]
d = {}
for p in passes:
    for t, v in p.items():
        d[t] = max(d.get(t, 0), v)
floors = {}
if hasattr(dc, "join_window_floors"):
    hw = [n for n in m.graph.node if is_hls_node(n) or is_rtl_node(n)]
    floors = dc.join_window_floors(m, hw)
bits = tot = 0
for n in m.graph.node:
    for i, t in enumerate(n.output):
        c = m.find_consumer(t)
        if c is None:
            continue
        tav = max(2, d.get(t, 0))
        dep = max(tav, floors.get(t, 0))
        try:
            w = getCustomOp(n).get_outstream_width(i)
        except Exception:
            w = 0
        tot += dep
        bits += dep * w
        if "-v" in sys.argv or t in floors:
            print(f"{t:28s} {n.name:34s} -> {c.name:34s} tav={tav:7d} floor={floors.get(t, 0):7d} "
                  f"depth={dep:7d} w={w}")
print(f"TOTAL depth {tot} kbit {bits / 1024.0:.1f} period {period}")
