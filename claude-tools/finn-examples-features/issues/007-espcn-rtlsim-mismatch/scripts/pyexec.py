import sys, numpy as np, time
from qonnx.core.modelwrapper import ModelWrapper
import finn.core.onnx_exec as oxe
G = "/home/lstasytis/finn/tests/benchmark/verification_io/espcn-bsd300_"
inp = np.load(G + "input.npy"); gold = np.load(G + "output.npy")
out_npz = sys.argv[2] if len(sys.argv) > 2 else None
m = ModelWrapper(sys.argv[1])
import os
from qonnx.custom_op.registry import getCustomOp
mode = os.environ.get("EXEC_MODE", "python")
for n in m.graph.node:
    try:
        getCustomOp(n).set_nodeattr("exec_mode", mode)
    except Exception:
        pass
iname = m.graph.input[0].name
ishape = tuple(m.get_tensor_shape(iname))
x = inp
if x.shape != ishape:
    x = x.transpose(0, 2, 3, 1) if ishape[-1] == 3 else x.reshape(ishape)
t = time.time()
ctx = oxe.execute_onnx(m, {iname: x}, return_full_exec_context=out_npz is not None)
o = ctx[m.graph.output[0].name]
print("exec %.1fs" % (time.time() - t), o.shape)
if o.shape != gold.shape:
    o = o.reshape(gold.shape) if o.size == gold.size else o
d = np.abs(o - gold)
print("max diff", d.max(), "frac bad", (d > 1e-6).mean(), "zeros", (o == 0).sum())
if out_npz:
    np.savez(out_npz, **{k: v for k, v in ctx.items() if isinstance(v, np.ndarray) and v.size > 1000})
