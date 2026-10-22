"""Per-node FINN resource estimates of a finished build's stitched model, as JSON.
    python node_est.py <build dir> <fpga part> <out.json>
Node attributes that drive the estimates are recorded too."""
import json
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

ATTRS = ["PE", "SIMD", "MW", "MH", "resType", "ram_style", "mem_mode", "ram_style_thresholds",
         "weightDataType", "inputDataType", "outputDataType", "accDataType", "binaryXnorMode",
         "noActivation", "depth", "impl_style", "Channels", "Kernel", "NumChannels",
         "numInputVectors", "IFMChannels", "ConvKernelDim", "IFMDim", "inWidth", "outWidth",
         "numSteps", "parallel_window", "depthwise"]

b, part, out = sys.argv[1:4]
m = ModelWrapper(b if b.endswith(".onnx") else f"{b}/intermediate_models/step_create_stitched_ip.onnx")
rows = []
for n in m.graph.node:
    inst = getCustomOp(n)
    est = inst.node_res_estimation(part)
    attrs = {}
    for a in ATTRS:
        try:
            v = inst.get_nodeattr(a)
            attrs[a] = v if isinstance(v, (int, float, str)) else list(v)
        except Exception:
            pass
    try:
        attrs["wmem"] = inst.calc_wmem()
    except Exception:
        pass
    try:
        attrs["tmem"] = inst.calc_tmem()
    except Exception:
        pass
    rows.append(dict(name=n.name, op=n.op_type, est=est, attrs=attrs))
json.dump(rows, open(out, "w"), indent=1, default=float)
