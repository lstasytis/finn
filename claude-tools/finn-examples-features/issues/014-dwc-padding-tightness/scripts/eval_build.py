"""Issue 014: estimate-only feature build (optimizer folding at the reference target, FIFOs
sized analytically, estimate reports) with a given padding, via the per-model test module.
Usage (worktree root, wtenv sourced):
  python eval_build.py <test dir> <board> <model or -> <padding> <seed> <out.json>"""
import importlib
import json
import os
import sys

import numpy as np

tdir, board, model, pad, seed, out = sys.argv[1:7]
pad, seed = int(pad), int(seed)
sys.path[:0] = [os.path.abspath("tests/benchmark"), os.path.abspath(tdir)]
mod = importlib.import_module(
    [f[:-3] for f in os.listdir(tdir) if f.startswith("test_build_") and f.endswith(".py")][0]
)
margs = () if model == "-" else (model,)
ref = mod.reference_estimate(board, *margs)
target_fps = int(mod.estimated_throughput(ref))
overrides = dict(
    mod.OPTIMIZER_FOLDING, target_fps=target_fps, folding_maximum_padding=pad, folding_seed=seed
)
m = mod.run_feature_build(False, board, *margs, f"e14p{pad}s{seed}", estimate_only=True, **overrides)

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402

d = m["output_dir"]


def load(p):
    with open(os.path.join(d, p)) as f:
        return json.load(f)


perf = load("report/estimate_network_performance.json")
layers = load("report/estimate_layer_resources.json")
tot, nofifo = {}, {}
for name, res in layers.items():
    if name == "total":
        continue
    for r, n in res.items():
        tot[r] = tot.get(r, 0) + n
        if not name.startswith("StreamingFIFO"):
            nofifo[r] = nofifo.get(r, 0) + n
BOUNDS = ["MW", "MH", "NumChannels", "Channels", "IFMChannels"]
start = ModelWrapper(os.path.join(ref, "intermediate_models/phase_convert_to_hardware.onnx"))
folded = ModelWrapper(os.path.join(d, "intermediate_models/step_target_fps_parallelization.onnx"))
before = {n.name: getCustomOp(n) for n in start.graph.node if n.domain.startswith("finn")}
padded = []
for n in folded.graph.node:
    if n.name in before and n.domain.startswith("finn"):
        a, b = getCustomOp(n), before[n.name]
        types = a.get_nodeattr_types()
        if any(x in types and a.get_nodeattr(x) != b.get_nodeattr(x) for x in BOUNDS):
            padded.append(n.name)
ref_perf = json.load(open(os.path.join(ref, "report/estimate_network_performance.json")))
ref_layers = json.load(open(os.path.join(ref, "report/estimate_layer_resources.json")))
ref_res = {}
for name, res in ref_layers.items():
    if name != "total" and not name.startswith("StreamingFIFO"):
        for r, n in res.items():
            ref_res[r] = ref_res.get(r, 0) + n
row = dict(
    model=model if model != "-" else os.path.basename(os.path.abspath(tdir)),
    board=board,
    pad=pad,
    seed=seed,
    target_fps=target_fps,
    ref_cycles=ref_perf["max_cycles"],
    ref_res=ref_res,
    cycles=perf["max_cycles"],
    est_fps=perf["estimated_throughput_fps"],
    res=tot,
    res_nofifo=nofifo,
    pad_dwcs=m.get("pad_dwcs"),
    padded_nodes=padded,
    fold_s=m.get("step_times_s", {}).get("step_target_fps_parallelization"),
    output_dir=d,
)
json.dump(row, open(out, "w"), indent=1, default=float)
print(json.dumps(row, default=float))
