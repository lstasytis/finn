# Issue 011: one Requant_rtl node of a build (rtlsim) vs the generic Requant's Python
# execution on the same input tensor (from the generic graph's Python reference).
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.core.onnx_exec as oxe

hw = ModelWrapper(sys.argv[1])
gen = ModelWrapper(sys.argv[2])
inp = np.load(sys.argv[3]).astype(np.float32)
hw_name, gen_name = sys.argv[4], sys.argv[5]
ref = oxe.execute_onnx(gen, {gen.graph.input[0].name: inp}, return_full_exec_context=True)
g = gen.get_node_from_name(gen_name)
n = hw.get_node_from_name(hw_name)
inst = getCustomOp(n)
inst.set_nodeattr("exec_mode", "rtlsim")
inst.prepare_rtlsim()
x = ref[g.input[0]]
print("scale", gen.get_initializer(g.input[1]).flatten()[:4], "bias", gen.get_initializer(g.input[2]).flatten()[:4])
print("hw scale", hw.get_initializer(n.input[1]).flatten()[:4], "hw bias", hw.get_initializer(n.input[2]).flatten()[:4])
ctx = {n.input[0]: x.reshape(inst.get_normal_input_shape(0))}
for i in n.input[1:]:
    ctx[i] = hw.get_initializer(i)
ctx[n.output[0]] = np.zeros(inst.get_normal_output_shape(), dtype=np.float32)
inst.execute_node(ctx, hw.graph)
y, r, xf = ctx[n.output[0]].flatten(), ref[g.output[0]].flatten(), x.flatten()
bad = np.nonzero(np.abs(y - r) > 1e-3)[0]
print(len(bad), "of", y.size, "differ")
for j in bad[:12]:
    print("in %g  ref %g  rtl %g" % (xf[j], r[j], y[j]))
print("in range of differing", xf[bad].min() if len(bad) else None, xf[bad].max() if len(bad) else None)
print("in range overall", xf.min(), xf.max())
