"""Size a characterized fixture and dump {tensor: {prod, cons, depth}} as JSON.

    python size_fixture.py <fixture_dir> <out.json> [module_path]

``module_path`` optionally names a scratch copy of derive_characteristic.py to use
instead of the worktree's (for ablations).
"""
import importlib.util
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.util.fpgadataflow import is_hls_node, is_rtl_node

fixture_dir, out_path = sys.argv[1:3]
if len(sys.argv) > 3:
    spec = importlib.util.spec_from_file_location("dc_variant", sys.argv[3])
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)
else:
    import finn.transformation.fpgadataflow.derive_characteristic as dc

with open(os.path.join(fixture_dir, "period.json")) as f:
    info = json.load(f)
m = ModelWrapper(os.path.join(fixture_dir, "characterized.onnx"))
m = m.transform(dc.DeriveFIFOSizes(period=info["period"], io_fifo_depth=info["io_fifo_depth"]))

out = {}
for n in m.graph.node:
    if not (is_hls_node(n) or is_rtl_node(n)):
        continue
    inst = getCustomOp(n)
    outd = inst.get_nodeattr("outFIFODepths")
    for i, t in enumerate(n.output):
        c = m.find_consumer(t)
        depth = outd[i]
        if c is not None:
            ci = getCustomOp(c)
            ind = ci.get_nodeattr("inFIFODepths")
            depth = max(depth, ind[list(c.input).index(t)])
        out[t] = {"prod": n.name, "cons": c.name if c is not None else None, "depth": int(depth)}
    ind = inst.get_nodeattr("inFIFODepths")
    for i, t in enumerate(n.input):
        if m.find_producer(t) is None and t in [x.name for x in m.graph.input]:
            out[t] = {"prod": None, "cons": n.name, "depth": int(ind[i])}
with open(out_path, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
tot = sum(v["depth"] for v in out.values())
print("%s: %d edges, total depth %d" % (fixture_dir, len(out), tot))
