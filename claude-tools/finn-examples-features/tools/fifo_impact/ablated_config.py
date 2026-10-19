"""Write the final_hw_config.json a build would get with an ablated sizer.

    python ablated_config.py <model_dir> <model|-> <board> <fixture_dir> <module.py> <out_dir>

Runs the real step_set_fifo_depths on the fixture's characterized model with the
sizer taken from <module.py>; the step writes <out_dir>/final_hw_config.json.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark"))
from feature_flow_helpers import configure, folding_config_file, load_bench_module  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402

import finn.builder.build_dataflow_config as build_cfg  # noqa: E402
import finn.builder.build_dataflow_steps as steps  # noqa: E402

model_dir, model, board, fixture_dir, module_path, out_dir = sys.argv[1:7]
model = None if model == "-" else model
os.makedirs(out_dir, exist_ok=True)
spec = importlib.util.spec_from_file_location("dc_variant", module_path)
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
steps.DeriveFIFOSizes = dc.DeriveFIFOSizes

mod = load_bench_module(model_dir)
cfg = configure(
    mod,
    model,
    board,
    out_dir,
    auto_fifo_depths=True,
    auto_fifo_strategy="heuristic_analytical",
    folding_config_file=folding_config_file(model_dir, model, board),
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
)
m = ModelWrapper(os.path.join(fixture_dir, "characterized.onnx"))
m = steps.step_set_fifo_depths(m, cfg)
m.save(os.path.join(out_dir, "sized.onnx"))
print(os.path.join(out_dir, "final_hw_config.json"))
