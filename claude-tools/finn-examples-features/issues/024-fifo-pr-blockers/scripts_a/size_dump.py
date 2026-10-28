"""Issue 024 A: DeriveFIFOSizes (of the worktree on PYTHONPATH) on a TAV model, dumped as JSON
{tensor: [producer, consumer, depth, width]}. usage: size_dump.py <tav.onnx> <out.json>"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.derive_characteristic import DeriveFIFOSizes
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
m = m.transform(DeriveFIFOSizes(period=int(m.get_metadata_prop("sizing_period"))))
out = {}
for n in m.graph.node:
    if not (is_hls_node(n) or is_rtl_node(n)):
        continue
    inst = getCustomOp(n)
    for i, t in enumerate(n.output):
        c = m.find_consumer(t)
        if c is not None:
            out[t] = [n.name, c.name, int(inst.get_nodeattr("outFIFODepths")[i]), int(inst.get_outstream_width(i))]
json.dump(out, open(sys.argv[2], "w"))
