import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
m = ModelWrapper(sys.argv[1])
for n in m.graph.node:
    i = getCustomOp(n)
    if n.op_type.startswith("StreamingFIFO"):
        print("  ", n.name, "depth", i.get_nodeattr("depth"), "w", i.get_instream_width())
    else:
        keys = ["SIMD", "PE", "parallel_window", "M", "IFMDim", "ConvKernelDim", "IFMChannels",
                "Stride", "MW", "MH", "numInputVectors", "inWidth", "outWidth", "shape"]
        a = {k: i.get_nodeattr(k) for k in keys if k in i.get_nodeattr_types()}
        print(n.name, a, "in", i.get_instream_width(), "out", i.get_outstream_width(),
              "exp", i.get_exp_cycles())
