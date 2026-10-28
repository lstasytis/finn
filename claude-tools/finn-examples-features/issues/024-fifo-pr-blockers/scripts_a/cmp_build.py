"""Issue 024 A: offline depths of a TAV model (DeriveFIFOSizes of the worktree on PYTHONPATH)
against a FIFO-inserted build model, per (producer, consumer). Prints differences and totals.
usage: cmp_build.py <tav.onnx> <build model with FIFOs> [-a]"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.derive_characteristic import DeriveFIFOSizes
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

m = ModelWrapper(sys.argv[1])
m = m.transform(DeriveFIFOSizes(period=int(m.get_metadata_prop("sizing_period"))))
off = {}
for n in m.graph.node:
    if not (is_hls_node(n) or is_rtl_node(n)):
        continue
    inst = getCustomOp(n)
    for i, t in enumerate(n.output):
        c = m.find_consumer(t)
        if c is not None:
            off[(n.name, c.name, t)] = (inst.get_nodeattr("outFIFODepths")[i], inst.get_outstream_width(i))
b = ModelWrapper(sys.argv[2])
built = {}
for n in b.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        p, c = b.find_producer(n.input[0]), b.find_consumer(n.output[0])
        if p and c:
            built[(p.name, c.name)] = built.get((p.name, c.name), []) + [getCustomOp(n).get_nodeattr("depth")]
tb = to = 0
for (p, c, t), (d, w) in off.items():
    bd = built.get((p, c), ["?"])
    to += d * w
    tb += (bd[0] if bd[0] != "?" else 0) * w
    if "-a" in sys.argv or d not in bd:
        print(f"{p:36s} -> {c:36s} offline {d:7d} build {bd}")
print(f"offline {to / 8192:.1f} kB  build {tb / 8192:.1f} kB")
