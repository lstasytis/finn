"""As node_est.py, but the optimizer's view: set_folding.node_resources with weight_info
(+-1 weights at one bit, constant weights scaled by their non-zero share).
    python node_est_opt.py <build dir> <fpga part> <out.json>"""
import json
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.set_folding import node_resources, weight_info

b, part, out = sys.argv[1:4]
m = ModelWrapper(f"{b}/intermediate_models/step_create_stitched_ip.onnx")
info = weight_info(m)
rows = [dict(name=n.name, op=n.op_type, est=node_resources(getCustomOp(n), part, info), attrs={})
        for n in m.graph.node]
json.dump(rows, open(out, "w"), indent=1, default=float)
