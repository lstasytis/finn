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
inst = getCustomOp(model.graph.node[0])
r = json.load(open(out))
# tree model of the current source (as patched / unpatched)
cum = inst.get_tree_model().cumulative(periods=2)
r["tree_in"] = cum[:, 0].tolist()
r["tree_out"] = cum[:, 1].tolist()
r["timing"] = list(inst.get_tree_model_timing())
json.dump(r, open(out.replace("m_", "n_"), "w"))
