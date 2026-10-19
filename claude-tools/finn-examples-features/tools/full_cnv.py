"""Full-mode (HLS + stitched-IP rtlsim) cnv-w1a1 optimizer folding, both FIFO sizings."""
import json, os, sys
sys.path.insert(0, "/tmp/finn_dev_lstasytis/worktrees/folding-fifo/tests/benchmark")
from feature_flow_helpers import run_flow
from test_folding_flow import folding_target
fps, budget = folding_target("bnn-pynq", "cnv-w1a1", "ZCU104", 10.0)
for strategy in ("heuristic_analytical", "largefifo_rtlsim"):
    r = run_flow("full_" + strategy, "bnn-pynq", "cnv-w1a1", "ZCU104", folding_config_file=None,
                 target_fps=fps, folding_style="optimizer", auto_fifo_strategy=strategy)
    print(json.dumps(dict(strategy=strategy, ok=r["build_ok"], verification=r["verification"], cycles=r["est_max_cycles"],
          budget=budget, rtlsim_fps=r["rtlsim_throughput_fps"], fifo_kbit=r.get("fifo_total_kbit"), res=r["est_resources"],
          fold_s=r["step_time_s"].get("step_target_fps_parallelization"), fifo_s=r["step_time_s"].get("step_set_fifo_depths"),
          total_s=r["total_time_s"], output_dir=r["output_dir"])), flush=True)
