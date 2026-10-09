"""ResNet50-w1a2 (finn-examples) through FINN's builder for the bitfile experiments.

Uses the finn-examples build of tests/benchmark/resnet50 (custom tidy/streamline/convert steps,
the U250 folding and specialize-layers configs; this FINN has no split_large_fifos) without its U250 SLR
floorplan step (Vitis only). Boards smaller than the U250 get a folding with every PE divided by
--pe-div (default 2 for the U55C, ~75 % of the U250's LUTs/DSPs/BRAMs; as for MobileNet,
folding_mobilenet_U250_halfpe.json), only where the divided PE still divides the channel count.

    frontend : up to step_set_fifo_depths (bitfiles with run_bitfile_experiment.py / run_vshell_timing.sh)

Usage:
    python run_resnet50.py --out <dir> [--clk 10] [--board U55C] [--pe-div 2] --mode frontend
"""

import argparse
import json
import os
import sys
import time

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import vitis_default_platform, vitis_part_map

BENCH = os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark")
sys.path.insert(0, os.path.join(BENCH, "resnet50"))
from custom_steps_resnet50 import (  # noqa: E402
    step_resnet50_convert_to_hw,
    step_resnet50_streamline,
    step_resnet50_tidy,
)

FRONTEND_STEPS = [
    step_resnet50_tidy,
    step_resnet50_streamline,
    step_resnet50_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]


def divided_folding(src, dst, div):
    """The U250 folding with every PE divided by div where that keeps it a divisor of the
    original PE (the original PE divides the channel count, so PE / div does too)."""
    cfg = json.load(open(src))
    n = 0
    for k, v in cfg.items():
        if isinstance(v, dict) and "PE" in v and v["PE"] % div == 0:
            v["PE"] //= div
            n += 1
    json.dump(cfg, open(dst, "w"), indent=2)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clk", type=float, default=10.0)
    ap.add_argument("--board", default="U55C")
    ap.add_argument("--pe-div", type=int, default=None, help="divide PE (default: 1 on U250, 2 otherwise)")
    ap.add_argument("--model", default=None, help="resnet50_w1a2_exported.onnx")
    ap.add_argument("--mode", choices=["frontend"], default="frontend")
    ap.add_argument("--stop-step", default=None)
    args = ap.parse_args()
    d = os.path.join(BENCH, "resnet50")
    front = os.path.join(args.out, "frontend")
    os.makedirs(front, exist_ok=True)
    div = args.pe_div or (1 if args.board == "U250" else 2)
    folding = os.path.join(d, "folding_config", "resnet50_folding_config_U250.json")
    if div > 1:
        dst = os.path.join(args.out, "folding_resnet50_pe_div%d.json" % div)
        print("PE divided by %d in %d layers" % (div, divided_folding(folding, dst, div)))
        folding = dst
    alveo = args.board in vitis_part_map
    kw = {"stop_step": args.stop_step} if args.stop_step else {}
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=front,
        steps=FRONTEND_STEPS,
        folding_config_file=folding,
        specialize_layers_config_file=os.path.join(
            d, "specialize_layers_config", "resnet50_specialize_layers_U250.json"
        ),
        auto_fifo_depths=False,
        synth_clk_period_ns=args.clk,
        board=args.board,
        shell_flow_type=(
            build_cfg.ShellFlowType.VITIS_ALVEO if alveo else build_cfg.ShellFlowType.VIVADO_ZYNQ
        ),
        vitis_platform=vitis_default_platform.get(args.board),
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        **kw,
    )
    t0 = time.time()
    model = args.model or os.path.join(
        os.environ["FINN_BUILD_DIR"], "rwr50", "models", "resnet50_w1a2_exported.onnx"
    )
    build.build_dataflow_cfg(model, cfg)
    res = {"mode": args.mode, "board": args.board, "pe_div": div, "total_s": time.time() - t0}
    json.dump(res, open(os.path.join(args.out, "%s.json" % args.mode), "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
