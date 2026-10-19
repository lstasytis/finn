# Does zeroing Thr2-output pixels reproduce the failing output's zero pattern?
import numpy as np, sys
from copy import deepcopy
from qonnx.core.modelwrapper import ModelWrapper
import finn.core.onnx_exec as oxe
S = sys.argv[1]
E = "/home/lstasytis/finn/build/finn_build/fx/fifo/est_espcn_KV260_SOM_uu6ftob8"
B = "/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_espcn_KV260_SOM_aiguk3l6"
g = ModelWrapper(E + "/intermediate_models/step_convert_to_hw.onnx")
ctx = np.load(S + "/ctx_hw.npz")
k0 = int(sys.argv[2])  # first generic node index to execute
nodes = list(g.graph.node)
x = ctx[nodes[k0].input[0]].copy()
mode = sys.argv[3]
fail = np.load(B + "/verification_output/verify_stitched_ip_rtlsim_0_FAIL.npy")
zero_fail = (fail == 0)[0, 0]
if mode == "thr2":  # NHWC 128x128
    for r in range(128):
        for a, b in [(93, 102), (117, 126)]:
            if r >= 37: x[0, r, a:b] = 0
        if r >= 38: x[0, r, 0:6] = 0
sub = deepcopy(g)
for n in nodes[:k0]:
    sub.graph.node.remove(n)
iname = nodes[k0].input[0]
vi = [v for v in g.graph.value_info if v.name == iname][0]
for v in list(sub.graph.value_info):
    if v.name == iname: sub.graph.value_info.remove(v)
while len(sub.graph.input): sub.graph.input.pop()
sub.graph.input.append(deepcopy(vi))
out = oxe.execute_onnx(sub, {iname: x})[g.graph.output[0].name]
z = (out == 0)[0, 0]
print("zeros sim", z.sum(), "zeros fail", zero_fail.sum(), "match", (z == zero_fail).mean(), "xor", (z ^ zero_fail).sum())
