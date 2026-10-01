"""MobileNetV1-w4a4 (finn-examples) through FINN's builder for the bitfile experiments.

Uses the finn-examples build configuration of tests/benchmark/mobilenet_v1: custom
streamlining/lowering steps, the board's folding and specialize-layers configs, standalone
thresholds on ZCU102/ZCU104 only (as in test_build_mobilenet_v1.py), FIFO sizing off.
Boards without their own config (U55C) use the U250 configs (the only Alveo ones).

    frontend : up to step_set_fifo_depths
    bitfile  : FINN's regular bitfile flow (Vitis for Alveo boards)
    islands  : Alveo: compute kernel built with the RapidWright island flow (rw_islands_pnr)

Zynq bitfiles are built with run_bitfile_experiment.py from the frontend's model.

Usage:
    python run_mobilenet.py --out <dir> [--clk 10] [--board ZCU104|U55C] --mode frontend
"""

import argparse
import json
import os
import shutil
import sys
import time

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import vitis_default_platform, vitis_part_map

BENCH = os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark")
sys.path.insert(0, os.path.join(BENCH, "mobilenet_v1"))
from custom_steps import step_mobilenet_lower_convs, step_mobilenet_streamline  # noqa: E402

FRONTEND_STEPS = [
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
BITFILE_STEPS = ["step_create_stitched_ip", "step_synthesize_bitfile"]


def config(args, out, steps, **kw):
    cfg_board = args.board if args.board in ("ZCU102", "ZCU104", "U250") else "U250"
    d = os.path.join(BENCH, "mobilenet_v1")
    alveo = args.board in vitis_part_map
    return build_cfg.DataflowBuildConfig(
        output_dir=out,
        steps=steps,
        folding_config_file=args.folding
        or os.path.join(d, "folding_config", "mobilenet_folding_config_%s.json" % cfg_board),
        specialize_layers_config_file=os.path.join(
            d, "specialize_layers_config", "mobilenet_specialize_layers_%s.json" % cfg_board
        ),
        synth_clk_period_ns=args.clk,
        board=args.board,
        shell_flow_type=(
            build_cfg.ShellFlowType.VITIS_ALVEO if alveo else build_cfg.ShellFlowType.VIVADO_ZYNQ
        ),
        vitis_platform=vitis_default_platform.get(args.board),
        auto_fifo_depths=False,
        standalone_thresholds=cfg_board in ("ZCU102", "ZCU104"),
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
        ],
        **kw,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clk", type=float, default=10.0)
    ap.add_argument("--board", default="ZCU104")
    ap.add_argument("--folding", default=None, help="folding config (default: finn-examples)")
    ap.add_argument("--mode", choices=["frontend", "bitfile", "islands"], default="frontend")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    front = os.path.join(args.out, "frontend")
    t0 = time.time()
    if args.mode == "frontend":
        model = os.path.join(BENCH, "models", "mobilenetv1-w4a4_pre_post_tidy_opset-11.onnx")
        build.build_dataflow_cfg(model, config(args, front, FRONTEND_STEPS))
    else:
        out = os.path.join(args.out, args.mode)
        os.makedirs(os.path.join(out, "intermediate_models"), exist_ok=True)
        src = os.path.join(front, "intermediate_models", "step_set_fifo_depths.onnx")
        shutil.copy(src, os.path.join(out, "intermediate_models", "step_set_fifo_depths.onnx"))
        kw = {"start_step": BITFILE_STEPS[0]}
        if args.mode == "islands":
            kw.update(rw_islands_pnr=True, dynarapid_workers=args.workers)
        build.build_dataflow_cfg(src, config(args, out, ["step_set_fifo_depths"] + BITFILE_STEPS, **kw))
    res = {"mode": args.mode, "board": args.board, "total_s": time.time() - t0}
    json.dump(res, open(os.path.join(args.out, "%s.json" % args.mode), "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
