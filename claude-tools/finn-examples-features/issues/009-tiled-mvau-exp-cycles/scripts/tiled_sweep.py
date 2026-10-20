"""Stitched-IP rtlsim steady-state cycles per frame of one MVAU_rtl config vs get_exp_cycles().
tiled_sweep.py MW MH SIMD PE TH NIV(comma) MEM IDT WDT [N_FRAMES]
MEM in internal_decoupled | external_mem | dynamic."""
import os
import sys
import time

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.rtlsim_exec import rtlsim_exec
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

PART = os.environ.get("SWEEP_PART", "xcvc1902-vsva2197-2MP-e-S")
CLK = float(os.environ.get("SWEEP_CLK", "5.0"))
a = sys.argv[1:]
mw, mh, simd, pe, th = map(int, a[:5])
niv = [int(x) for x in a[5].split(",")]
mem = a[6]
idt, wdt = DataType[a[7]], DataType[a[8]]
nfr = int(a[9]) if len(a) > 9 else 4
tag = "_".join(a[:9]).replace(",", "x") + os.environ.get("SWEEP_TAG", "")

W = gen_finn_dt_tensor(wdt, (mw, mh))
inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, niv + [mw])
outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, niv + [mh])
node = helper.make_node(
    "MVAU", ["inp", "weights"], ["outp"], domain="finn.custom_op.fpgadataflow",
    backend="fpgadataflow", MW=mw, MH=mh, SIMD=simd, PE=pe, TH=th,
    inputDataType=idt.name, weightDataType=wdt.name, outputDataType="INT32",
    ActVal=0, binaryXnorMode=0, noActivation=1, numInputVectors=niv,
    mem_mode=mem, resType="dsp", preferred_impl_style="rtl",
)
if mem == "dynamic":
    node.input[1] = "weights_in"
    win = helper.make_tensor_value_info("weights_in", TensorProto.FLOAT, [mw, mh])
    g = helper.make_graph([node], "g", [inp, win], [outp])
else:
    g = helper.make_graph([node], "g", [inp], [outp])
m = ModelWrapper(qonnx_make_model(g, producer_name="i009"))
m.set_tensor_datatype("inp", idt)
m.set_tensor_datatype("outp", DataType["INT32"])
if mem == "dynamic":
    m.set_tensor_datatype("weights_in", wdt)
else:
    m.set_tensor_datatype("weights", wdt)
    m.set_initializer("weights", W)
m = m.transform(SpecializeLayers(PART))
m = m.transform(GiveUniqueNodeNames())
assert m.graph.node[0].op_type == "MVAU_rtl", m.graph.node[0].op_type
m = m.transform(MinimizeAccumulatorWidth())
m = m.transform(InferDataTypes())
inst = getCustomOp(m.graph.node[0])
exp = inst.get_exp_cycles()
comp = inst.get_exp_compute_cycles()
fetch = inst.get_exp_weight_fetch_cycles()
t0 = time.time()
m = m.transform(PrepareIP(PART, CLK))
m = m.transform(HLSSynthIP())
m = m.transform(CreateStitchedIP(PART, CLK))
t1 = time.time()
if mem == "external_mem":
    # cppxsi (dummy data) has no AXI-MM master model: use the real-data rtlsim
    # with 1 and 1+nfr frames and take the difference
    x = gen_finn_dt_tensor(idt, [1] + niv[1:] + [mw])

    def cyc(n):
        ctx = {"inp": np.concatenate([x] * n)}
        rtlsim_exec(m, ctx)
        return int(m.get_metadata_prop("cycles_rtlsim"))

    c1 = cyc(1)
    cn = cyc(1 + nfr)
    ss = (cn - c1) / nfr
    lat = c1
else:
    d = xsi_fifosim(m, nfr, max_iters=max(20000, 4 * exp * nfr), behav=False)
    ss = d["steady_state_cycles"] / max(1, d["steady_state_frames"])
    lat = d["latency_cycles"]
    assert d["TIMEOUT"] == 0, d
t2 = time.time()
print(
    "RESULT", tag, "exp", exp, "compute", comp, "fetch", fetch, "rtlsim", ss,
    "ratio %.4f" % (ss / exp), "latency", lat, "frames", nfr,
    "ipgen_s %.0f sim_s %.0f" % (t1 - t0, t2 - t1), flush=True,
)
