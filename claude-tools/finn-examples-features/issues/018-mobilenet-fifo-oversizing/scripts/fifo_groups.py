"""Issue 018: write depth JSONs for groups of FIFOs of a copied build, using the TAV model's
drives-pacer flags. Groups: thr2 (every FIFO into a throttleable consumer at 2), top<N>_2 (the
N largest in bits at 2), nonthr2 (FIFOs into pacer-driving consumers at 2).
usage: fifo_groups.py <copy dir> <tav.onnx> <out prefix>"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

cm, tm, pre = ModelWrapper(sys.argv[1] + "/step_hw_ipgen.onnx"), ModelWrapper(sys.argv[2]), sys.argv[3]
gp = int(tm.get_metadata_prop("sizing_period"))
nodes = [n for n in tm.graph.node if is_hls_node(n) or is_rtl_node(n)]
tavs = dc.read_node_tavs(tm, nodes)
per = dc.node_periods(nodes, tavs)
gp = max(gp, max(per.values()))
dp = dc.pacer_drivers(tm, nodes, per, gp * dc.PACER_TOLERANCE)
rows = []
for node in cm.graph.node:
    if not node.op_type.startswith("StreamingFIFO"):
        continue
    inst = getCustomOp(node)
    c = cm.find_consumer(node.output[0])
    p = cm.find_producer(node.input[0])
    if c is None or p is None:
        continue
    rows.append((inst.get_nodeattr("depth") * inst.get_instream_width(), node.name, dp[c.name]))
groups = {
    "thr2": {n: 2 for _, n, d in rows if not d},
    "nonthr2": {n: 2 for _, n, d in rows if d},
}
for k in (15, 30):
    groups[f"top{k}_2"] = {n: 2 for _, n, _ in sorted(rows, reverse=True)[:k]}
for g, v in groups.items():
    json.dump(v, open(f"{pre}_{g}.json", "w"), indent=0)
    print(g, len(v))
