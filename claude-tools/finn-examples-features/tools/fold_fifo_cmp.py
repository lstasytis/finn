import os, sys
os.environ["FEATURE_FLOW_QUICK"] = "1"
sys.path.insert(0, "/home/lstasytis/finn_dev_lstasytis/finn/tests/benchmark")
from feature_flow_helpers import run_flow
from test_folding_flow import folding_target
for model_dir, model in [("bnn-pynq", "cnv-w1a1"), ("bnn-pynq", "cnv-w2a2"), ("bnn-pynq", "tfc-w2a2")]:
    fps, budget = folding_target(model_dir, model, "ZCU104", 10.0)
    for name, extra in [("naive", dict(folding_style="naive")),
                        ("optimizer", dict(folding_style="optimizer")),
                        ("optimizer+fifo", dict(folding_style="optimizer", enable_folding_fifo_heuristic=True))]:
        r = run_flow("foldfifo_" + name.replace("+", "_"), model_dir, model, "ZCU104", folding_config_file=None,
                     target_fps=fps, auto_fifo_strategy="heuristic_analytical", **extra)
        ok = r["build_ok"] and all(r["verification"].get(s) for s in r["verify_steps"])
        print("%-9s %-15s ok=%s cycles=%s/%d fifo_kbit=%.1f fold=%.1fs LUT=%s BRAM=%s" % (
            model, name, ok, r["est_max_cycles"], budget, r["fifo_total_kbit"] or 0,
            r["step_time_s"].get("step_target_fps_parallelization", 0),
            r["est_resources"].get("LUT"), r["est_resources"].get("BRAM_18K")), flush=True)
