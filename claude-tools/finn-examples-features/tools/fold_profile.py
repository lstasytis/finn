"""Time the stages of one SetFolding run on a cached convert checkpoint (tooling only).

    python fold_profile.py <est build dir> <board> [target_cycles|max] [padding] [out.json]

Prints wall time per stage (probe, cheapest guess, PE/SIMD annealing, resource-type
annealing, repair+polish, whole-graph scorings) and the result's cycles / resources.
"""
import json
import sys
import time
from collections import defaultdict
from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.set_folding as sf
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance

est, board = sys.argv[1], sys.argv[2]
target = sys.argv[3] if len(sys.argv) > 3 else None
padding = int(sys.argv[4]) if len(sys.argv) > 4 else 0
out = sys.argv[5] if len(sys.argv) > 5 else None
if target in (None, "ref"):
    target = json.load(open(f"{est}/report/estimate_network_performance.json"))["max_cycles"]
target = None if target == "max" else int(target)

T = defaultdict(float)
N = defaultdict(int)


def timed(cls, name, key=None):
    fn = getattr(cls, name)

    def wrap(*a, **kw):
        k = key(a, kw) if key else name
        t = time.time()
        try:
            return fn(*a, **kw)
        finally:
            T[k] += time.time() - t
            N[k] += 1

    setattr(cls, name, wrap)


def opt_key(a, kw):
    tp = kw.get("target_parameters", ("SIMD", "PE"))
    return "optimize[" + ",".join(tp)[:24] + "]"


timed(sf.Optimizer, "optimize", opt_key)
timed(sf.Optimizer, "cheapest_guess")
timed(sf.Optimizer, "cost_model")
timed(sf.SetFolding, "_probe_fastest_throughput")
timed(sf.SetFolding, "_slowest_cycles")
timed(sf.SetFolding, "_refine")
timed(sf.SetFolding, "_score")
timed(sf.SetFolding, "_verify_as_built")
timed(sf.SetFolding, "_fold_at")

model = ModelWrapper(f"{est}/intermediate_models/phase_convert_to_hardware.onnx")
n_hw = len(model.graph.node)
t0 = time.time()
folder = sf.SetFolding(
    target, platform=board, folding_effort=50, seed=1, folding_maximum_padding=padding
)
model = model.transform(folder, apply_to_subgraphs=True)
total = time.time() - t0
perf = model.analysis(dataflow_performance)
res = {
    "est": est,
    "target": target,
    "padding": padding,
    "nodes": n_hw,
    "total_s": round(total, 1),
    "max_cycles": perf["max_cycles"],
    "report": {
        k: v for k, v in (folder.last_search_report or {}).items() if not isinstance(v, dict)
    },
    "stages": {k: (round(T[k], 1), N[k]) for k in sorted(T, key=lambda k: -T[k])},
}
print(json.dumps(res, indent=1, default=str))
if out:
    json.dump(res, open(out, "w"), indent=1, default=str)
