"""Issue 018: a bounded MON=1 perf.py result: interval, storage, and the FIFOs whose peak
occupancy reached their depth (saturated: fifo.sv reports up to depth + 3).
usage: sat.py <copy dir> <mon.json> [-a]"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

m = ModelWrapper(sys.argv[1] + "/step_hw_ipgen.onnx")
d = json.load(open(sys.argv[2]))
print("interval", d["interval_cycles"], "kB", d["bits"] / 8192, "changed", len(d["changed"]))
for n in m.graph.node:
    if not n.op_type.startswith("StreamingFIFO"):
        continue
    dep = d["changed"].get(n.name, [None, getCustomOp(n).get_nodeattr("depth")])[1]
    pk = d["peaks"][n.name]
    if (pk >= dep and dep > 2) or "-a" in sys.argv or (dep == 2 and pk >= 5):
        p, c = m.find_producer(n.input[0]), m.find_consumer(n.output[0])
        print(f"{n.name:22s} {p.name if p else '-':34s} -> {c.name if c else '-':34s} "
              f"depth {dep:6d} peak {pk:6d} w={getCustomOp(n).get_instream_width()}"
              f"{' CHANGED' if n.name in d['changed'] else ''}")
