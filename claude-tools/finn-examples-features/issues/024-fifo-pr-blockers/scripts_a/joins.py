"""Issue 024 A: per join (multi-input node) of a FIFO-inserted model: each input's FIFO depth,
its folded frame words and the producer chain up to the FIFO. usage: joins.py <model.onnx>"""
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])
for n in m.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        continue
    ins = [t for t in n.input if m.find_producer(t) is not None]
    if len(ins) < 2:
        continue
    inst = getCustomOp(n)
    print(n.name)
    for i, t in enumerate(n.input):
        f = m.find_producer(t)
        if f is None:
            continue
        fw = int(np.prod(inst.get_folded_input_shape(i)[:-1]))
        if f.op_type.startswith("StreamingFIFO"):
            d = getCustomOp(f).get_nodeattr("depth")
            p = m.find_producer(f.input[0])
        else:
            d, p = "-", f
        print(f"   in{i} depth={d} frame_words={fw} from {p.name if p else '-'}")
