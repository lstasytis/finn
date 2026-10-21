"""Issue 015 (4b): rtlsim TAV characterization (heuristic_rtlsim's per-node step) of single
nodes of a build, to find the one that times out. Works on a copy of the build's
step_hw_ipgen.onnx (node IP must exist). usage: chrc_rtlsim.py <build dir> <op_type prefix> ...  ('' = every characterized node)"""
import sys
import traceback

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.derive_characteristic import (
    _UNCHARACTERIZED_OPS,
    is_stream_join,
)

part, clk = "xczu7ev-ffvc1156-2-e", 10.0
m = ModelWrapper(sys.argv[1] + "/intermediate_models/step_hw_ipgen.onnx")
for n in m.graph.node:
    if not n.op_type.startswith(tuple(sys.argv[2:])):
        continue
    if n.op_type in _UNCHARACTERIZED_OPS or is_stream_join(n):
        continue
    inst = getCustomOp(n)
    try:
        inst.derive_token_access_vectors(m, inst.get_exp_cycles() + 20, "rtlsim", part, clk,
                                         n.op_type)
        print(n.name, "OK period", inst.get_nodeattr("io_chrc_period"), flush=True)
    except Exception as e:
        print(n.name, "FAIL", str(e).splitlines()[0][:200], flush=True)
        traceback.print_exc()
