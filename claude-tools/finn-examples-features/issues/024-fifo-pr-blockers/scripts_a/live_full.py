"""Issue 024 A: map a maxcounts_live.txt dump (rtlsim_xsi_live.cpp) to the FIFOs of the model
(graph order of StreamingFIFO_rtl) and list FIFOs (depth > 2) whose peak reached their implemented capacity.
usage: live_full.py <step_hw_ipgen.onnx> <maxcounts_live.txt> [all]"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import math


def capacity(depth, width, style="auto"):
    """DEPTH_ACTUAL of finn-rtllib/fifo/hdl/fifo.sv"""
    if depth <= 33:
        eff = "shift"
    elif style != "auto":
        eff = style
    elif depth <= 64 and width < 12:
        eff = "shift"
    elif depth <= 257 or width < 5:
        eff = "distributed"
    elif depth <= 2028:
        eff = "block"
    else:
        eff = "ultra"
    if eff == "shift":
        return max(depth, 5)
    if eff == "distributed":
        return depth
    uram = eff == "ultra"
    min_abits, q = (12, 16) if uram else (9, 0)
    req = depth - q - 1
    lo, hi = math.ceil(math.log2(req)), 0
    if lo > min_abits:
        h = math.ceil(math.log2(req - 2 ** (lo - 1)))
        if h < lo - 1:
            lo, hi = lo - 1, max(h, 1)
    impl = 2**lo + (2**hi if hi else 0)
    pipe = 3 + (2**lo - 1) // 8192 if uram else 2
    return impl + pipe + q + 1


m = ModelWrapper(sys.argv[1])
lines = open(sys.argv[2]).read().split("\n")
print(lines[0])
mc = dict(ln.split("\t") for ln in lines[1:] if "\t" in ln)
for ind, n in enumerate(m.get_nodes_by_op_type("StreamingFIFO_rtl")):
    key = "maxcount" if ind == 0 else "maxcount_%d" % ind
    inst = getCustomOp(n)
    d = inst.get_nodeattr("depth")
    cap = capacity(d, inst.get_instream_width_padded(), inst.get_nodeattr("ram_style"))
    pk = int(mc.get(key, -1))
    p, c = m.find_producer(n.input[0]), m.find_consumer(n.output[0])
    if (pk >= cap and d > 2) or "all" in sys.argv:
        print(f"{n.name:22s} {p.name if p else '-':34s} -> {c.name if c else '-':34s} "
              f"depth {d:7d} capacity {cap:7d} peak {pk}")
