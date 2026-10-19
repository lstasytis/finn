import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])
tot_bits = 0
for n in m.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        i = getCustomOp(n)
        d = i.get_nodeattr("depth"); w = i.get_instream_width()
        p = m.find_producer(n.input[0]); c = m.find_consumer(n.output[0])
        tot_bits += d * w
        print("%-40s %-40s depth=%6d width=%4d" % (p.name if p else "<in>", c.name if c else "<out>", d, w))
print("total kbit", tot_bits / 1024.0)
