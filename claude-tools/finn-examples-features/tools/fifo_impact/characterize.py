"""Capture a characterized fixture: the model with TAVs and the period handed to
DeriveFIFOSizes, by running the real step_set_fifo_depths with the sizer intercepted.

    python characterize.py <model_dir> <model|-> <board> <quick_run_dir> <fixture_dir>

<quick_run_dir> is a feature-flow run (quick mode) holding
intermediate_models/step_transpose_decomposition.onnx, the input of step_set_fifo_depths.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark"))
from feature_flow_helpers import configure, folding_config_file, load_bench_module  # noqa: E402
from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.util.basic import get_by_name  # noqa: E402

import finn.builder.build_dataflow_config as build_cfg  # noqa: E402
import finn.builder.build_dataflow_steps as steps  # noqa: E402
from finn.transformation.fpgadataflow.derive_characteristic import DeriveFIFOSizes  # noqa: E402

model_dir, model, board, run_dir, fixture_dir = sys.argv[1:6]
model = None if model == "-" else model
os.makedirs(fixture_dir, exist_ok=True)


class Capture(DeriveFIFOSizes):
    def apply(self, m):
        # the step unlinks the TAV sidecars afterwards: keep private copies
        tav_dir = os.path.join(fixture_dir, "tavs")
        os.makedirs(tav_dir, exist_ok=True)
        originals = []
        for n in m.graph.node:
            for attr in ("io_chrc_in", "io_chrc_out"):
                a = get_by_name(n.attribute, attr)
                if a is None:
                    continue
                path = a.s.decode("utf-8")
                if path.endswith(".npy") and os.path.isfile(path):
                    dst = os.path.join(tav_dir, n.name + "_" + attr + ".npy")
                    shutil.copy(path, dst)
                    originals.append((a, a.s))
                    a.s = dst.encode("utf-8")
        m.save(os.path.join(fixture_dir, "characterized.onnx"))
        for a, s in originals:
            a.s = s
        with open(os.path.join(fixture_dir, "period.json"), "w") as f:
            json.dump({"period": int(self.period), "io_fifo_depth": self.io_fifo_depth}, f)
        return super().apply(m)


steps.DeriveFIFOSizes = Capture
mod = load_bench_module(model_dir)
cfg = configure(
    mod,
    model,
    board,
    fixture_dir,
    auto_fifo_depths=True,
    auto_fifo_strategy="heuristic_analytical",
    folding_config_file=folding_config_file(model_dir, model, board),
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
)
m = ModelWrapper(os.path.join(run_dir, "intermediate_models", "step_transpose_decomposition.onnx"))
m = steps.step_set_fifo_depths(m, cfg)
m.save(os.path.join(fixture_dir, "sized_by_step.onnx"))
print("fixture written to", fixture_dir)
