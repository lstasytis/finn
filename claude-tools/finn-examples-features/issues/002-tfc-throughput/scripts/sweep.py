"""Standalone stitched-IP interval of one HLS node config.
sweep.py mvau MW MH SIMD PE IDT WDT ACT|none   or   sweep.py ls LABELS PE K IDT"""
import sys

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import calculate_signed_dot_prod_range, gen_finn_dt_tensor

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from fpgadataflow.test_fpgadataflow_labelselect import make_labelselect_modelwrapper
from fpgadataflow.test_fpgadataflow_mvau import make_single_fclayer_modelwrapper

PART = "xczu3eg-sfvc784-2-e"
a = sys.argv[1:]
if a[0] == "mvau":
    mw, mh, simd, pe = map(int, a[1:5])
    idt, wdt = DataType[a[5]], DataType[a[6]]
    act = None if a[7] == "none" else DataType[a[7]]
    W = gen_finn_dt_tensor(wdt, (mw, mh))
    if act is None:
        T = None
        tdt = None
        odt = DataType["UINT32"] if wdt == DataType["BIPOLAR"] else DataType["INT32"]
    else:
        odt = act
        (lo, hi) = calculate_signed_dot_prod_range(idt, wdt, mw)
        T = np.sort(np.random.randint(lo, hi - 1, (mh, act.get_num_possible_values() - 1)), 1)
        T = T.astype(np.float32)
        if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
            tdt = DataType["UINT32"]
            T = np.ceil((T + mw) / 2)
        else:
            tdt = DataType["INT32"]
    m = make_single_fclayer_modelwrapper(W, pe, simd, wdt, idt, odt, T, tdt)
    for n in m.graph.node:
        getCustomOp(n).set_nodeattr("mem_mode", "internal_embedded")
        getCustomOp(n).set_nodeattr("preferred_impl_style", "hls")
else:
    labels, pe, k = map(int, a[1:4])
    m = make_labelselect_modelwrapper(labels, pe, k, DataType[a[4]], "hls")
m = m.transform(SpecializeLayers(PART))
m = m.transform(GiveUniqueNodeNames())
m = m.transform(PrepareIP(PART, 10.0))
m = m.transform(HLSSynthIP())
m = m.transform(CreateStitchedIP(PART, 10.0))
exp = list(m.analysis(exp_cycles_per_layer).values())[0]
d = xsi_fifosim(m, 16, max_iters=50000, behav=False)
ss = d["steady_state_cycles"] / max(1, d["steady_state_frames"])
print("RESULT", " ".join(a), "exp", exp, "interval", ss, "excess", ss - exp, flush=True)
