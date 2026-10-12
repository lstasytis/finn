"""Node-by-node rtlsim of the chain SWG ... Pool of a model (original and fixed by
fix_pool_swg_simd.py) against FINN's python execution of the same chain (base ops, no folding).

    python check_pool_fix.py <model.onnx> <first node> <last node> <work dir> [part] [clk_ns]
"""

import os
import sys

import numpy as np
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode


def sub_model(model, first, last):
    names = [n.name for n in model.graph.node]
    nodes = model.graph.node[names.index(first) : names.index(last) + 1]
    tin, tout = nodes[0].input[0], nodes[-1].output[0]
    vi = {v.name: v for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
    g = helper.make_graph(
        [n for n in nodes], "chain", [vi[tin]], [vi[tout]],
        value_info=[vi[n.output[0]] for n in nodes[:-1] if n.output[0] in vi])
    m = ModelWrapper(qonnx_make_model(g, producer_name="check"))
    for t in [tin] + [n.output[0] for n in nodes]:
        m.set_tensor_datatype(t, model.get_tensor_datatype(t))
    for n in m.graph.node:
        inst = getCustomOp(n)
        for a in ("code_gen_dir_ipgen", "ipgen_path", "ip_path", "ip_vlnv", "rtlsim_so"):
            if a in inst.get_nodeattr_types():
                inst.set_nodeattr(a, "")
    return m


def python_ref(m):
    """The same chain with the base ops (python execution, layout independent)."""
    r = ModelWrapper(m.model.__deepcopy__())
    for n in r.graph.node:
        for suf in ("_rtl", "_hls"):
            if n.op_type.endswith(suf):
                n.op_type = n.op_type[: -len(suf)]
        n.domain = "finn.custom_op.fpgadataflow"
    r = r.transform(SetExecMode("python"))
    return r


if __name__ == "__main__":
    model, first, last, work = ModelWrapper(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    part = sys.argv[5] if len(sys.argv) > 5 else "xcu55c-fsvh2892-2L-e"
    clk = float(sys.argv[6]) if len(sys.argv) > 6 else 3.333
    os.makedirs(work, exist_ok=True)
    os.environ["FINN_BUILD_DIR"] = work
    m = sub_model(model, first, last)
    tin = m.graph.input[0].name
    x = gen_finn_dt_tensor(m.get_tensor_datatype(tin), m.get_tensor_shape(tin))
    ref = execute_onnx(python_ref(m), {tin: x})[m.graph.output[0].name]
    m = m.transform(PrepareIP(part, clk)).transform(HLSSynthIP()).transform(SetExecMode("rtlsim")).transform(PrepareRTLSim())
    out = execute_onnx(m, {tin: x})[m.graph.output[0].name]
    out = out.reshape(ref.shape)
    ok = np.array_equal(out, ref)
    np.save(os.path.join(work, "out.npy"), out)
    np.save(os.path.join(work, "ref.npy"), ref)
    np.save(os.path.join(work, "x.npy"), x)
    d = (out - ref).flatten()
    print("DIFF max abs %s, values %s" % (np.abs(d).max(), dict(zip(*np.unique(d, return_counts=True)))))
    print("RESULT %s %s..%s match=%s mismatching=%d/%d" % (sys.argv[1], first, last, ok,
          int(np.sum(out != ref)), ref.size))
