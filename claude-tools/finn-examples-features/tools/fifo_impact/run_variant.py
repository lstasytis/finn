"""Full-mode feature flow run with FIFO depths from a config JSON (no sizing).

    python run_variant.py <flow_name> <model_dir> <model|-> <board> <config.json>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark"))
from feature_flow_helpers import run_flow  # noqa: E402

flow, model_dir, model, board, config = sys.argv[1:6]
model = None if model == "-" else model
row = run_flow(flow, model_dir, model, board, auto_fifo_depths=False, folding_config_file=config)
print(
    "RESULT", flow, row["model"], row["build_ok"], row["verification"],
    row.get("rtlsim_throughput_fps"), row.get("fifo_total_kbit"), row["output_dir"],
)
