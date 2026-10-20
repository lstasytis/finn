# Issue 011: depths of the FIFOs on fork (DuplicateStreams/Split) outputs and join
# (ElementwiseAdd/Concat/dynamic MVAU) inputs, with the frame size in words.
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])


def walk(t, fwd):
    # skip through FIFOs/DWCs to the next real node
    depth, path = [], []
    while True:
        n = m.find_consumer(t) if fwd else m.find_producer(t)
        if n is None:
            return None, depth, path
        if n.op_type.startswith("StreamingFIFO"):
            depth.append(getCustomOp(n).get_nodeattr("depth"))
        elif not n.op_type.startswith("StreamingDataWidthConverter"):
            return n, depth, path
        path.append(n.name)
        t = n.output[0] if fwd else n.input[0]


for n in m.graph.node:
    if n.op_type.startswith(("DuplicateStreams", "StreamingSplit")):
        for o in n.output:
            c, d, _ = walk(o, True)
            words = np.prod(getCustomOp(n).get_folded_output_shape()[:-1])
            print("%-24s -> %-24s fifo %s (frame %d words)" % (n.name, c.name if c else None, d, words))
    if n.op_type.startswith(("ElementwiseAdd", "StreamingConcat")) or (
        n.op_type.startswith("MVAU") and getCustomOp(n).get_nodeattr("mem_mode") == "dynamic"
    ):
        for i in n.input[:2]:
            p, d, _ = walk(i, False)
            print("%-24s <- %-24s fifo %s" % (n.name, p.name if p else None, d))
