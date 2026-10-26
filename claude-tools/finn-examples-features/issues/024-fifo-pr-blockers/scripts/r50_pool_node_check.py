"""Node-level check of resnet50 (V80, committed full folding) global average pool:
ConvolutionInputGenerator_rtl_21 (depthwise, SIMD 2048) -> DWC 12288->24 bit -> Pool_hls_1 (PE 4).
Runs per-node rtlsim of (a) the chain as built and (b) the same chain with SWG SIMD = Pool PE = 4
(no DWC) on one random INT6 input and compares both with a numpy average pool.
Run one variant per process (finn_xsi keeps a loaded simulation per top-module name).
    python r50_pool_node_check.py <final model .onnx> <fpgapart> asbuilt|fixed"""
import copy
import numpy as np
import sys
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

src = ModelWrapper(sys.argv[1])
part = sys.argv[2]
names = ["ConvolutionInputGenerator_rtl_21", "StreamingDataWidthConverter_rtl_153", "Pool_hls_1"]
CODEGEN = (
    "code_gen_dir_ipgen",
    "ipgen_path",
    "ip_path",
    "ip_vlnv",
    "rtlsim_so",
    "code_gen_dir_cppsim",
)


def build(fixed):
    nodes = [copy.deepcopy(src.get_node_from_name(n)) for n in names]
    if fixed:
        nodes = [nodes[0], nodes[2]]
    for a, b in zip(nodes[:-1], nodes[1:]):
        b.input[0] = a.output[0]
    vis = []
    for n in nodes:
        for t in list(n.input) + list(n.output):
            vis.append(helper.make_tensor_value_info(t, 1, src.get_tensor_shape(t)))
    vis = {v.name: v for v in vis}
    g = helper.make_graph(
        nodes,
        "r50pool",
        [vis[nodes[0].input[0]]],
        [vis[nodes[-1].output[0]]],
        value_info=[v for k, v in vis.items() if k not in (nodes[0].input[0], nodes[-1].output[0])],
    )
    m = ModelWrapper(helper.make_model(g, producer_name="r50pool"))
    nodes = list(m.graph.node)  # make_graph copies the protos: edit the model's own nodes
    for n in nodes:
        for t in list(n.input) + list(n.output):
            m.set_tensor_datatype(t, src.get_tensor_datatype(t))
        inst = getCustomOp(n)
        for a in CODEGEN:
            if a in inst.get_nodeattr_types():
                inst.set_nodeattr(a, "")
    if fixed:
        getCustomOp(nodes[0]).set_nodeattr("SIMD", getCustomOp(nodes[1]).get_nodeattr("PE"))
    m = m.transform(SetExecMode("rtlsim"))
    m = m.transform(PrepareIP(part, 4.0))
    m = m.transform(HLSSynthIP())
    m = m.transform(PrepareRTLSim())
    return m


rng = np.random.default_rng(0)
x = rng.integers(-32, 32, size=(1, 7, 7, 2048)).astype(np.float32)
s = x.reshape(49, 2048).sum(0)
res = {}
for fixed in (sys.argv[3] == "fixed",):
    m = build(fixed)
    y = execute_onnx(m, {m.graph.input[0].name: x})[m.graph.output[0].name].reshape(-1)
    res[fixed] = y
    print(
        "variant",
        "fixed (SWG SIMD 4, no DWC)" if fixed else "as built (SWG SIMD 2048 -> DWC -> PE 4)",
    )
    for sh in range(4, 10):
        ref = np.clip(np.floor(s / 2**sh), -8, 7)
        print(
            "   vs clip(floor(channel sum / 2^%d)): %d of 2048 differ" % (sh, int((ref != y).sum()))
        )
    # what the as-built chain sums: 49 consecutive 4-channel words of the SWG stream (kernel-major)
    stream = x.reshape(49 * 512, 4)  # SWG SIMD 2048 order: k, then channels
    wrong = stream.reshape(512, 49, 4).sum(1).reshape(-1)
    for sh in range(4, 10):
        ref = np.clip(np.floor(wrong / 2**sh), -8, 7)
        print(
            "   vs the k-major mis-grouped sums / 2^%d: %d of 2048 differ"
            % (sh, int((ref != y).sum()))
        )
np.save("r50pool_%s.npy" % sys.argv[3], y)
