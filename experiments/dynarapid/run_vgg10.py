"""VGG10 (RadioML, finn-examples) bitfile on the ZCU104: Vivado vs DynaRapid.

Uses the finn-examples build configuration of tests/benchmark/vgg10-radioml (custom steps,
folding and specialize-layers config, 4 ns). The front end (up to FIFO sizing) runs once;
both bitfile flows start from its result:

    frontend : step_tidy_up .. step_set_fifo_depths
    vivado   : step_create_stitched_ip, step_synthesize_bitfile (regular ZynqBuild)
    dynarapid: step_synthesize_bitfile with dynarapid_pnr=True

Usage:
    python run_vgg10.py --model radioml_w4a4_small_tidy.onnx --out <dir> \
        --mode frontend|vivado|dynarapid [--library <libdir>]
"""

import argparse
import json
import os
import shutil
import sys
import time

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

BENCH = os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark", "vgg10-radioml")
sys.path.insert(0, BENCH)
from custom_steps import step_convert_final_layers, step_pre_streamline  # noqa: E402

FRONTEND_STEPS = [
    "step_tidy_up",
    step_pre_streamline,
    "step_streamline",
    "step_convert_to_hw",
    step_convert_final_layers,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]
BACKEND_STEPS = {
    "vivado": ["step_create_stitched_ip", "step_synthesize_bitfile"],
    "dynarapid": ["step_synthesize_bitfile"],
    "islands": ["step_synthesize_bitfile"],
}


def shell_flow(board):
    """Alveo boards (U55C, U250, ...) use the Vitis flow, the others the Zynq flow."""
    from finn.util.basic import vitis_part_map

    if board in vitis_part_map:
        return build_cfg.ShellFlowType.VITIS_ALVEO
    return build_cfg.ShellFlowType.VIVADO_ZYNQ


def config(out, steps, board, clk=4.0, **kw):
    return build_cfg.DataflowBuildConfig(
        output_dir=out,
        steps=steps,
        folding_config_file=BENCH + "/folding_config/vgg10radioml_folding_config.json",
        specialize_layers_config_file=BENCH
        + "/specialize_layers_config/vgg10radioml_specialize_layers.json",
        synth_clk_period_ns=clk,
        board=board,
        shell_flow_type=shell_flow(board),
        standalone_thresholds=True,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
        ],
        **kw,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["frontend", "vivado", "dynarapid", "islands"], required=True)
    ap.add_argument("--library", default=None)
    ap.add_argument("--board", default="ZCU104")
    ap.add_argument("--clk", type=float, default=4.0)
    args = ap.parse_args()
    front = os.path.join(args.out, "frontend")
    t0 = time.time()
    if args.mode == "frontend":
        build.build_dataflow_cfg(args.model, config(front, FRONTEND_STEPS, args.board, args.clk))
    else:
        out = os.path.join(args.out, args.mode)
        # start from the front end's FIFO-sized model
        os.makedirs(os.path.join(out, "intermediate_models"), exist_ok=True)
        src = os.path.join(front, "intermediate_models", "step_set_fifo_depths.onnx")
        steps = ["step_set_fifo_depths"] + BACKEND_STEPS[args.mode]
        shutil.copy(src, os.path.join(out, "intermediate_models", "step_set_fifo_depths.onnx"))
        kw = {"start_step": BACKEND_STEPS[args.mode][0]}
        if args.mode == "dynarapid":
            kw.update(dynarapid_pnr=True, dynarapid_library_dir=args.library)
        if args.mode == "islands":
            # Alveo: compute kernel built with the RapidWright island flow
            kw.update(rw_islands_pnr=True)
        build.build_dataflow_cfg(src, config(out, steps, args.board, args.clk, **kw))
    res = {"mode": args.mode, "total_s": time.time() - t0}
    with open(os.path.join(args.out, "vgg10_%s.json" % args.mode), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
