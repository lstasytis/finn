"""Issue 018: depth JSON capping the FIFOs into throttleable consumers (TAV model's flags) at C,
optionally only those whose consumer op type does (KEEP=SWG keeps FIFOs into
ConvolutionInputGenerator unchanged).
usage: cap_group.py <copy dir> <tav.onnx> <C> <out.json> [KEEP=<op prefix>]"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

cm, tm, cap, out = (ModelWrapper(sys.argv[1] + "/step_hw_ipgen.onnx"), ModelWrapper(sys.argv[2]),
                    int(sys.argv[3]), sys.argv[4])
keep = sys.argv[5].split("=")[1] if len(sys.argv) > 5 else None
gp = int(tm.get_metadata_prop("sizing_period"))
nodes = [n for n in tm.graph.node if is_hls_node(n) or is_rtl_node(n)]
per = dc.node_periods(nodes, dc.read_node_tavs(tm, nodes))
gp = max(gp, max(per.values()))
dp = dc.pacer_drivers(tm, nodes, per, gp * dc.PACER_TOLERANCE)
res = {}
for n in cm.graph.node:
    if not n.op_type.startswith("StreamingFIFO"):
        continue
    c = cm.find_consumer(n.output[0])
    if c is None or cm.find_producer(n.input[0]) is None or dp[c.name]:
        continue
    if keep and c.op_type.startswith(keep):
        continue
    d = getCustomOp(n).get_nodeattr("depth")
    if d > cap:
        res[n.name] = cap
json.dump(res, open(out, "w"), indent=0)
print("capped", len(res))
