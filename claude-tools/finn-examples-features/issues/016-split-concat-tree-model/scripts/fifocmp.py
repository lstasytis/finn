import glob, sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
def fifos(d):
    m = ModelWrapper(glob.glob(d + "/intermediate_models/step_set_fifo_depths.onnx")[0])
    r = {}
    for n in m.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            p = m.find_producer(n.input[0]); c = m.find_consumer(n.output[0])
            key = (p.name if p else "IN", c.name if c else "OUT")
            i = getCustomOp(n)
            r[key] = (i.get_nodeattr("depth"), i.get_instream_width())
    return r
a = fifos(glob.glob(sys.argv[1])[0]); b = fifos(glob.glob(sys.argv[2])[0])
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print(k, a.get(k), b.get(k))
print("total kbit", sum(d*w for d, w in a.values())/1024, sum(d*w for d, w in b.values())/1024)
