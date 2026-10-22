"""Issue 018: a MON=1 perf.py result against the copied build: per FIFO sized depth, rtlsim
peak occupancy, width; totals of the sized and of the peak+1 depths.
usage: peaks.py <copy dir> <mon.json> [N]"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1] + "/step_hw_ipgen.onnx")
pk = json.load(open(sys.argv[2]))["peaks"]
nshow = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rows = []
for node in m.graph.node:
    if not node.op_type.startswith("StreamingFIFO"):
        continue
    i = getCustomOp(node)
    p, c = m.find_producer(node.input[0]), m.find_consumer(node.output[0])
    d, w = i.get_nodeattr("depth"), i.get_instream_width()
    rows.append((d * w, node.name, p.name if p else "-", c.name if c else "-", d, pk[node.name], w))
sb = sum(r[0] for r in rows)
pb = sum(max(2, r[5] + 1) * r[6] for r in rows)
print(f"sized {sb / 8192:.1f} kB, peak+1 {pb / 8192:.1f} kB")
for r in sorted(rows, reverse=True)[:nshow]:
    print(f"{r[1]:22s} {r[2]:34s} -> {r[3]:34s} sized {r[4]:6d} peak {r[5]:6d} w={r[6]}")
