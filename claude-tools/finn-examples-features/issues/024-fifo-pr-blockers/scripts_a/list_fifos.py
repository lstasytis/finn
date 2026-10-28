"""Issue 018: list a build's FIFOs (producer -> consumer, depth, width, bits), largest first.
usage: list_fifos.py <model.onnx> [N]"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
rows = []
for node in m.graph.node:
    if not node.op_type.startswith("StreamingFIFO"):
        continue
    inst = getCustomOp(node)
    d, w = inst.get_nodeattr("depth"), inst.get_instream_width()
    p, c = m.find_producer(node.input[0]), m.find_consumer(node.output[0])
    rows.append((d * w, node.name, p.name if p else "-", c.name if c else "-", d, w))
tot = sum(r[0] for r in rows)
print(f"{len(rows)} FIFOs, {tot / 8192:.1f} kB, {tot / 1024:.1f} kbit")
acc = 0
for r in sorted(rows, reverse=True)[:n]:
    acc += r[0]
    print(f"{r[1]:22s} {r[2]:36s} -> {r[3]:36s} d={r[4]:6d} w={r[5]:5d} {r[0] / 8192:7.2f} kB "
          f"cum {100 * acc / tot:5.1f}%")
