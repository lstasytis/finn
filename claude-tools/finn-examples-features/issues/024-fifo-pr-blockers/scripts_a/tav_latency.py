"""Issue 024 A: first/last write time (wall clock, chained-TAV schedule, one pass) of every
node output, to compare the expected first-output latency with rtlsim.
usage: tav_latency.py <tav.onnx> [node ...]"""
import sys

import finn.transformation.fpgadataflow.derive_characteristic as dc
from qonnx.core.modelwrapper import ModelWrapper

m = ModelWrapper(sys.argv[1])
gp = int(m.get_metadata_prop("sizing_period"))
orig = dc.Edge
rec = {}


def edge(writes, per_frame, *a):
    f = sys._getframe(1).f_locals
    node = f.get("node")
    t = f.get("t")
    if node is not None and t is not None:
        rec[(node.name, t)] = (int(writes[0]), int(writes[per_frame - 1]), per_frame)
    return orig(writes, per_frame, *a)


dc.Edge = edge
dc.derive_chained_tav_depths(m, gp, True, True)
want = sys.argv[2:]
print("period", gp)
for (n, t), (a, b, pf) in rec.items():
    if not want or n in want:
        print(f"{n:36s} first {a:11d} ({a / gp:6.2f} gp) last of frame 0 {b:11d} ({b / gp:6.2f} gp)")
