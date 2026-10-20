"""Model first-frame latency of every node of a saved TAV model: the time of the last output
token of the first frame with all inputs available at cycle 0 (local clock = TAV clock), after
the causal pass (phase lead, causal writes). Compare with nodesim.py's rtlsim "latency".
usage: tav_latency.py <tav.onnx>"""
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
    except Exception:
        continue
    if ri is None or not oc:
        continue
    streams = {t: c for t, c in ic.items() if m.find_producer(t) is not None or
               t in [x.name for x in m.graph.input]}
    span = len(ri[0])
    clock = np.arange(span, dtype=np.int64)
    scheds = {t: (np.zeros(int(c[-1]), dtype=np.int64), c) for t, c in streams.items()}
    t, c = next(iter(oc.items()))
    w = dc.token_times(clock, c)
    lead = dc.phase_lead(c, list(streams.values()))
    wc = dc.causal_writes(w, scheds, lead)
    pf = dc.frame_tokens(np.asarray(c))
    print(f"{n.name:36s} period={span // 2:6d} pf={pf:5d} lead={lead:4d} "
          f"first_w={int(w[0]):6d} frame1_done={int(wc[pf-1]):6d}")
