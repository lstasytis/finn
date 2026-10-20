"""Per op type of a saved TAV model: the output tokens each node's TAV writes before its first
read on a streamed input (phase lead), and the tokens per frame. usage: tav_lead.py <tav.onnx>"""
import sys
from collections import defaultdict

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
res = defaultdict(list)
for n in m.graph.node:
    if not (is_hls_node(n) or is_rtl_node(n)):
        continue
    try:
        inst = getCustomOp(n)
        ri, ro = dc.read_tav_rows(inst)
        ic, oc = dc.stream_curves(n, inst, ri, ro)
    except Exception:
        continue
    if ri is None:
        continue
    streams = [c for t, c in ic.items() if m.find_producer(t) is not None or t in
               [x.name for x in m.graph.input]]
    for t, c in oc.items():
        lead = dc.phase_lead(c, streams) if streams else 0
        res[n.op_type].append((lead, dc.frame_tokens(np.asarray(c))))
for k, v in sorted(res.items()):
    leads = [x for x, _ in v]
    print(f"{k:34s} nodes={len(v):3d} with_lead={sum(1 for x in leads if x):3d} "
          f"max_lead={max(leads):6d} e.g. {[x for x in v if x[0]][:3]}")
