"""Measure node-level rtlsim schedules of StreamingSplit_hls / StreamingConcat_hls.

usage: measure.py <split|concat> <chans,comma> <simd> <nvec,comma> <pragma 0|1> <out.json>
Traces every stream over 5 frames (as derive_token_access_vectors_using_rtlsim does)
and stores the raw per-cycle handshake traces plus the current tree model's TAVs.
"""
import json
import sys

import numpy as np
from onnx import TensorProto
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hls.concat_hls import StreamingConcat_hls
from finn.custom_op.fpgadataflow.hls.split_hls import StreamingSplit_hls
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

op, chans, simd, nvec, pragma, out = sys.argv[1:7]
chans = [int(x) for x in chans.split(",")]
simd = int(simd)
nvec = [int(x) for x in nvec.split(",")]
pragma = int(pragma)
PRAG = "#pragma HLS dataflow disable_start_propagation"


def patch(cls):
    orig = cls.pragmas

    def pragmas(self):
        orig(self)
        lst = self.code_gen_dict["$PRAGMAS$"]
        lst[:] = [p for p in lst if p != PRAG]
        if pragma:
            lst.insert(0, PRAG)

    cls.pragmas = pragmas


patch(StreamingSplit_hls)
patch(StreamingConcat_hls)

part = "xc7z020clg400-1"
dt = DataType["INT8"]
tot = sum(chans)
if op == "split":
    inp = [oh.make_tensor_value_info("in0", TensorProto.FLOAT, nvec + [tot])]
    outs = [
        oh.make_tensor_value_info("out%d" % i, TensorProto.FLOAT, nvec + [c])
        for i, c in enumerate(chans)
    ]
    node = oh.make_node(
        "StreamingSplit",
        ["in0"],
        ["out%d" % i for i in range(len(chans))],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=simd,
        ChannelsPerStream=chans,
        inputDataType=dt.name,
        numInputVectors=nvec,
        preferred_impl_style="hls",
    )
else:
    inp = [
        oh.make_tensor_value_info("in%d" % i, TensorProto.FLOAT, nvec + [c])
        for i, c in enumerate(chans)
    ]
    outs = [oh.make_tensor_value_info("out0", TensorProto.FLOAT, nvec + [tot])]
    node = oh.make_node(
        "StreamingConcat",
        ["in%d" % i for i in range(len(chans))],
        ["out0"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        SIMD=simd,
        ChannelsPerStream=chans,
        inputDataTypes=[dt.name] * len(chans),
        numInputVectors=nvec,
        preferred_impl_style="hls",
    )
model = ModelWrapper(qonnx_make_model(oh.make_graph([node], "g", inp, outs)))
for t in inp + outs:
    model.set_tensor_datatype(t.name, dt)
model = model.transform(SpecializeLayers(part))
model = model.transform(GiveUniqueNodeNames())
model = model.transform(PrepareIP(part, 4))
model = model.transform(HLSSynthIP())
model = model.transform(PrepareRTLSim())
inst = getCustomOp(model.graph.node[0])

P = 5
n_vec = int(np.prod(nvec))
if op == "split":
    n_in = n_vec * tot // simd
    io_dict = {
        "inputs": {"in0": [0] * (n_in * P)},
        "outputs": {"out%d" % i: [] for i in range(len(chans))},
    }
else:
    io_dict = {
        "inputs": {"in%d" % i: [0] * (n_vec * c // simd * P) for i, c in enumerate(chans)},
        "outputs": {"out0": []},
    }
sim = inst.get_rtlsim()
inst.reset_rtlsim(sim)
names = list(io_dict["inputs"].keys()) + list(io_dict["outputs"].keys())
tr = {k: sim.trace_stream(k + "_V") for k in names}
try:
    inst.rtlsim_multi_io(sim, io_dict, sname="_V", batch_size=P)
finally:
    inst.close_rtlsim(sim)
cycles = inst.get_nodeattr("cycles_rtlsim")
traces = {k: str(v) for k, v in tr.items()}

# tree model of the current source (as patched / unpatched)
cum = inst.get_tree_model().cumulative(periods=2)
res = dict(
    op=op,
    chans=chans,
    simd=simd,
    nvec=nvec,
    pragma=pragma,
    cycles=int(cycles),
    traces=traces,
    tree_in=cum[:, 0].tolist(),
    tree_out=cum[:, 1].tolist(),
)
json.dump(res, open(out, "w"))
print("done", op, chans, simd, nvec, pragma, cycles)
