"""MobileNetV1-w4a4 (finn-examples) front end for the bitfile experiments.

Uses the finn-examples build configuration of tests/benchmark/mobilenet_v1 (custom
streamlining/lowering steps, ZCU104 folding; there is no ZCU104 specialize-layers config, so
FINN's defaults apply). Runs up to step_set_fifo_depths (FIFO sizing off, as in finn-examples);
the resulting dataflow model is the input of run_bitfile_experiment.py (vivado / islands).

Usage:
    python run_mobilenet.py --out <dir> [--clk 10] [--board ZCU104]
"""

import argparse
import json
import os
import sys
import time

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

BENCH = os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark")
sys.path.insert(0, os.path.join(BENCH, "mobilenet_v1"))
from custom_steps import step_mobilenet_lower_convs, step_mobilenet_streamline  # noqa: E402

STEPS = [
    step_mobilenet_streamline,
    step_mobilenet_lower_convs,
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_transpose_decomposition",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clk", type=float, default=10.0)
    ap.add_argument("--board", default="ZCU104")
    args = ap.parse_args()
    model = os.path.join(BENCH, "models", "mobilenetv1-w4a4_pre_post_tidy_opset-11.onnx")
    fold = os.path.join(
        BENCH, "mobilenet_v1", "folding_config", "mobilenet_folding_config_%s.json" % args.board
    )
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=os.path.join(args.out, "frontend"),
        steps=STEPS,
        folding_config_file=fold,
        synth_clk_period_ns=args.clk,
        board=args.board,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        auto_fifo_depths=False,
        standalone_thresholds=True,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
    )
    t0 = time.time()
    build.build_dataflow_cfg(model, cfg)
    res = {"mode": "frontend", "total_s": time.time() - t0}
    json.dump(res, open(os.path.join(args.out, "frontend.json"), "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
