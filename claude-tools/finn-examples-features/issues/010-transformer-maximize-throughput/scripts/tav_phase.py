"""Per node of a saved TAV model (tav_model.py): TAV length, first read / first write cycle and
tokens per frame of each stream. usage: tav_phase.py <tav.onnx>"""
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
for n in m.graph.node:
    if not (is_hls_node(n) or is_rtl_node(n)):
        continue
    try:
        inst = getCustomOp(n)
        ri, ro = dc.read_tav_rows(inst)
        ic, oc = dc.stream_curves(n, inst, ri, ro)
    except Exception as e:
        print(f"{n.name:36s} no TAV ({type(e).__name__})")
        continue
    if ri is None:
        print(f"{n.name:36s} no TAV")
        continue
    s = []
    for tag, curves in (("in", ic), ("out", oc)):
        for t, c in curves.items():
            c = np.asarray(c)
            nz = np.nonzero(c)[0]
            s.append(f"{tag}:{t[:22]} first={nz[0] if len(nz) else -1} pf={dc.frame_tokens(c)}")
    print(f"{n.name:36s} len={len(ri)} " + " | ".join(s))
