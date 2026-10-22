"""Issue 018: depths of an eval_rule.py variant (or of the sizer on PYTHONPATH with 'none'),
written as {FIFO name: depth} for a copied build (prep_copy.py), matched by producer/consumer.
usage: variant_depths.py <tav.onnx> <variant|none> <copy dir> <out.json>"""
import json
import sys

from qonnx.core.modelwrapper import ModelWrapper

sys.argv, args = sys.argv[:2] + ["none"], sys.argv[1:]
import eval_rule as er  # noqa: E402  (runs its own baseline on import: argv patched)

tav, variant, cdir, out = args
fn = er.orig if variant == "none" else er.VARIANTS[variant]
sizes = er.size(tav, fn)
by_pc = {(p, c): d for p, c, d, _ in sizes.values()}
cm = ModelWrapper(cdir + "/step_hw_ipgen.onnx")
res = {}
for n in cm.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        p, c = cm.find_producer(n.input[0]), cm.find_consumer(n.output[0])
        if p is not None and c is not None and (p.name, c.name) in by_pc:
            res[n.name] = max(2, by_pc[(p.name, c.name)])
json.dump(res, open(out, "w"), indent=0)
print("FIFOs", len(res))
