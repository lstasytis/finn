"""Per-FIFO comparison of the main sweep's largefifo_rtlsim vs heuristic_analytical runs.
    python compare_sweep.py <results_dir> <model> <board>"""
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

res_dir, model, board = sys.argv[1:4]


def latest(flow):
    found = None
    with open(os.path.join(res_dir, flow + ".jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if (r["model"], r["board"]) == (model, board) and r["build_ok"] and not r.get("quick"):
                found = r
    return found


def fifos(row):
    m = ModelWrapper(os.path.join(row["output_dir"], "intermediate_models", "step_set_fifo_depths.onnx"))
    out = {}
    for n in m.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            i = getCustomOp(n)
            p = m.find_producer(n.input[0]); c = m.find_consumer(n.output[0])
            key = ((p.name if p else "<in>"), (c.name if c else "<out>"))
            out[key] = (i.get_nodeattr("depth"), i.get_instream_width(), i.get_nodeattr("impl_style") if "impl_style" in i.get_nodeattr_types() else "")
    return out


rows = {f: latest(f) for f in ("fifo_largefifo_rtlsim", "fifo_heuristic_analytical")}
for f, r in rows.items():
    print(f, r["rtlsim_throughput_fps"], r["fifo_total_kbit"], r["output_dir"])
a, b = fifos(rows["fifo_largefifo_rtlsim"]), fifos(rows["fifo_heuristic_analytical"])
print("%-40s %-40s %8s %8s %6s" % ("producer", "consumer", "largefifo", "analytic", "width"))
for key in sorted(set(a) | set(b), key=lambda k: (k[0], k[1])):
    da = a.get(key, ("-", "-", ""))
    db = b.get(key, ("-", "-", ""))
    print("%-40s %-40s %8s %8s %6s" % (key[0], key[1], da[0], db[0], da[1] if da[1] != "-" else db[1]))
