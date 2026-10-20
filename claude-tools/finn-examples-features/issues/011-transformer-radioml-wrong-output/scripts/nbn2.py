# Issue 011: node-by-node rtlsim of a build's ipgen'd dataflow graph, compared with
# the Python execution of the generic-HW graph (est build, step_convert_to_hw).
# Compute nodes are matched by op base type and occurrence order; FIFOs/DWCs are
# executed but not compared. Inputs are reshaped to each node's normal input shape.
# Usage: nbn2.py <hw_ipgen.onnx> <generic.onnx> <input.npy>
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

hw = ModelWrapper(sys.argv[1]).transform(SetExecMode("rtlsim"))
hw = hw.transform(PrepareRTLSim(num_workers=8))
gen = ModelWrapper(sys.argv[2])
inp = np.load(sys.argv[3]).astype(np.float32)
ref = oxe.execute_onnx(gen, {gen.graph.input[0].name: inp}, return_full_exec_context=True)


def base(op):
    for suf in ("_hls", "_rtl"):
        op = op.replace(suf, "")
    return op


gen_by_type = {}
for n in gen.graph.node:
    gen_by_type.setdefault(base(n.op_type), []).append(n)
seen = {}
ctx = {hw.graph.input[0].name: inp}
for init in hw.graph.initializer:
    ctx[init.name] = hw.get_initializer(init.name)
n_div = 0
for k, node in enumerate(hw.graph.node):
    inst = getCustomOp(node)
    for i, name in enumerate(node.input):
        if name in ctx and hw.get_initializer(name) is None:
            shp = tuple(inst.get_normal_input_shape(i))
            if ctx[name].shape != shp and np.prod(ctx[name].shape) == np.prod(shp):
                ctx[name] = ctx[name].reshape(shp)
    inst.execute_node(ctx, hw.graph)
    b = base(node.op_type)
    if b in ("StreamingFIFO", "StreamingDataWidthConverter"):
        continue
    idx = seen.get(b, 0)
    seen[b] = idx + 1
    g = gen_by_type[b][idx]
    for j, o in enumerate(node.output):
        out, r = ctx[o].reshape(-1), ref[g.output[j]].reshape(-1)
        d = np.abs(out - r).max() if out.size == r.size else float("inf")
        ok = d <= 1e-3
        n_div += not ok
        print(
            "%3d %-28s ~ %-22s out%d %s maxdiff %.4g (%d/%d elems differ)"
            % (k, node.name, g.name, j, "ok" if ok else "DIVERGES", d,
               int((np.abs(out - r) > 1e-3).sum()) if out.size == r.size else -1, r.size),
            flush=True,
        )
    if n_div >= 4:
        print("stopping after 4 diverging tensors")
        break
