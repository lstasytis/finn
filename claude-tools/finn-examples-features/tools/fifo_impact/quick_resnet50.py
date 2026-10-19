"""Quick-mode FIFO sizing run of resnet50 (V80 config) with the Vivado version
check bypassed: nothing is synthesized in quick mode."""
import os
import sys

sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark"))
import finn.builder.build_dataflow_checks as checks  # noqa: E402

checks.get_vivado_version = lambda: (2024, 2)
from feature_flow_helpers import run_flow  # noqa: E402

os.environ["FEATURE_FLOW_QUICK"] = "1"
os.environ["FEATURE_FLOW_VERIFY"] = ""
row = run_flow(
    "fifo_heuristic_analytical", "resnet50", None, "V80",
    auto_fifo_depths=True, auto_fifo_strategy="heuristic_analytical",
)
print(row["build_ok"], row["output_dir"], row.get("fifo_total_depth"))
