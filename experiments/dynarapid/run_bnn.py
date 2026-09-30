"""BNN-PYNQ (finn-examples) models through FINN's regular builder, for quick bitstream tests.

Uses the finn-examples bnn-pynq configs (tests/benchmark/bnn-pynq), optionally with every PE
and SIMD set to 1 (`--fold1`, the smallest possible design, for fast functional tests).
FIFO sizing is off by default: the BNN-PYNQ topologies are linear chains, which cannot
deadlock with shallow FIFOs (only throughput suffers).

    frontend : up to step_hw_ipgen (+ FIFO insertion)
    bitfile  : FINN's regular bitfile flow (Vitis for Alveo boards, ZynqBuild otherwise)

Usage:
    python run_bnn.py --model cnv-w1a1 --board U55C --clk 10 --fold1 --out <dir> \
        --mode frontend|bitfile
"""

import argparse
import json
import os
import shutil
import time

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import vitis_default_platform, vitis_part_map

BENCH = os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark")
HERE = os.path.dirname(os.path.abspath(__file__))

FRONTEND_STEPS = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
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
BITFILE_STEPS = ["step_create_stitched_ip", "step_synthesize_bitfile"]


def config(args, out, steps, **kw):
    alveo = args.board in vitis_part_map
    fold = os.path.join(BENCH, "bnn-pynq", "folding_config", "%s_folding_config.json" % args.model)
    if args.fold1:
        fold = os.path.join(HERE, "folding_%s_pe1simd1.json" % args.model.replace("-", "_"))
    return build_cfg.DataflowBuildConfig(
        output_dir=out,
        steps=steps,
        folding_config_file=fold,
        specialize_layers_config_file=os.path.join(
            BENCH, "bnn-pynq", "specialize_layers_config", "%s_specialize_layers.json" % args.model
        ),
        synth_clk_period_ns=args.clk,
        board=args.board,
        shell_flow_type=(
            build_cfg.ShellFlowType.VITIS_ALVEO if alveo else build_cfg.ShellFlowType.VIVADO_ZYNQ
        ),
        vitis_platform=vitis_default_platform.get(args.board),
        auto_fifo_depths=args.fifo_sizing,
        stitched_ip_gen_dcp=False,
        default_swg_exception=True,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
        ],
        **kw,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnv-w1a1")
    ap.add_argument("--board", default="U55C")
    ap.add_argument("--clk", type=float, default=10.0)
    ap.add_argument("--fold1", action="store_true", help="all PE/SIMD = 1")
    ap.add_argument("--fifo-sizing", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["frontend", "bitfile", "islands"], required=True)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    front = os.path.join(args.out, "frontend")
    t0 = time.time()
    if args.mode == "frontend":
        onnx = os.path.join(BENCH, "models", args.model + ".onnx")
        build.build_dataflow_cfg(onnx, config(args, front, FRONTEND_STEPS))
    else:
        out = os.path.join(args.out, args.mode)
        os.makedirs(os.path.join(out, "intermediate_models"), exist_ok=True)
        src = os.path.join(front, "intermediate_models", "step_set_fifo_depths.onnx")
        shutil.copy(src, os.path.join(out, "intermediate_models", "step_set_fifo_depths.onnx"))
        steps = ["step_set_fifo_depths"] + BITFILE_STEPS
        kw = {"start_step": BITFILE_STEPS[0]}
        if args.mode == "islands":
            kw.update(rw_islands_pnr=True, dynarapid_workers=args.workers)
        build.build_dataflow_cfg(src, config(args, out, steps, **kw))
    res = {"model": args.model, "board": args.board, "mode": args.mode, "total_s": time.time() - t0}
    with open(os.path.join(args.out, "%s.json" % args.mode), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
