"""Issue 018: offline FIFO sizing of a saved TAV model (013's tav_model.py) by running
DeriveFIFOSizes itself (whatever the worktree on PYTHONPATH implements) and reading the
depths it writes. Prints every edge (-v) and the totals.
usage: size_offline.py <tav.onnx> [-v]"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.derive_characteristic import DeriveFIFOSizes
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
period = int(m.get_metadata_prop("sizing_period"))
m = m.transform(DeriveFIFOSizes(period=period))
bits = tot = 0
for n in m.graph.node:
    if not (is_hls_node(n) or is_rtl_node(n)):
        continue
    inst = getCustomOp(n)
    for i, t in enumerate(n.output):
        c = m.find_consumer(t)
        if c is None:
            continue
        dep = inst.get_nodeattr("outFIFODepths")[i]
        try:
            w = inst.get_outstream_width(i)
        except Exception:
            w = 0
        tot += dep
        bits += dep * w
        if "-v" in sys.argv:
            print(f"{t:28s} {n.name:34s} -> {c.name:34s} depth={dep:7d} w={w}")
print(f"TOTAL depth {tot} kbit {bits / 1024.0:.1f} period {period}")
