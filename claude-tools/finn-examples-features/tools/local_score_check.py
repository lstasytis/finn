"""Check SetFolding's local (per node/edge) scoring against the whole-graph _score on every
refine evaluation of one fold (tooling only).

    python local_score_check.py <est build dir> <board> [target_cycles|ref] [padding]
"""
import json
import sys
from qonnx.core.modelwrapper import ModelWrapper

import finn.transformation.fpgadataflow.set_folding as sf

est, board = sys.argv[1], sys.argv[2]
target = sys.argv[3] if len(sys.argv) > 3 else "ref"
padding = int(sys.argv[4]) if len(sys.argv) > 4 else 0
if target == "ref":
    target = json.load(open(f"{est}/report/estimate_network_performance.json"))["max_cycles"]
stats = {"evals": 0, "bad": 0}
orig = sf.SetFolding._local_scorer


def checked(self, opt):
    local = orig(self, opt)
    if local is None:
        print("local scorer not applicable")
        return None

    def score():
        names, achieved, metrics = local()
        scored, w_achieved, w_metrics = self._score(opt.model, opt)
        w_names = self._bottleneck_nodes(scored)
        stats["evals"] += 1
        diff = {
            r: (metrics.get(r, 0), w_metrics.get(r, 0))
            for r in set(metrics) | set(w_metrics)
            if abs(metrics.get(r, 0) - w_metrics.get(r, 0)) > 1e-9 * max(1, abs(w_metrics.get(r, 0)))
        }
        if achieved != w_achieved or names != w_names or diff:
            stats["bad"] += 1
            if stats["bad"] <= 5:
                print("MISMATCH", achieved, w_achieved, names ^ w_names, diff)
        return names, achieved, metrics

    return score


sf.SetFolding._local_scorer = checked
model = ModelWrapper(f"{est}/intermediate_models/phase_convert_to_hardware.onnx")
folder = sf.SetFolding(
    int(target), platform=board, folding_effort=50, seed=1, folding_maximum_padding=padding
)
model.transform(folder, apply_to_subgraphs=True)
print("RESULT", est.rsplit("/", 1)[-1], "padding", padding, stats)
