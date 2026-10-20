"""Estimate-only maximize folding of a transformer model (optimizer, fixed seed), like
test_transformer_folding[maximize] without the hardware build. Prints the metrics and the
slowest nodes. usage: fold_max_est.py [model] [board]; run from a worktree with wtenv.sh."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark"))
sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark", "transformer"))
from test_build_transformer import OPTIMIZER_FOLDING, run_feature_build  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "transformer_benchmark"
board = sys.argv[2] if len(sys.argv) > 2 else "ZCU104"
m = run_feature_build(board, model, "fold-max-est", estimate_only=True, **OPTIMIZER_FOLDING)
print("METRICS", json.dumps(m))
cyc = json.load(open(m["output_dir"] + "/report/estimate_layer_cycles.json"))
for k, v in sorted(cyc.items(), key=lambda x: -x[1])[:12]:
    print("NODE", k, v)
