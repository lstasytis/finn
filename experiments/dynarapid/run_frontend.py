"""Frontend (model -> folded, FIFO-sized dataflow model) of a FINN benchmark model, for the
bitfile experiments (run_bitfile_experiment.py / run_campaign.sh).

Uses the upstream benchmark suite's own build configuration (tests/benchmark/<bench>/
test_build_*.py configure_build: custom steps, folding and specialize-layers configs, standalone
thresholds, ...) and only overrides what the experiments need:
- the clock (--clk), the target board (--board; --config-board selects whose folding config is
  used, e.g. ResNet50's V80 config on the U55C),
- the parallelism: --target-fps (FINN's folding search instead of the hand-written PE/SIMD,
  scaling_folding.py) or --pe-div (every PE of the config divided, where it stays a divisor),
- FIFO sizing (--fifo, default heuristic_analytical: tree models, no IP generation / rtlsim),
- the build stops after phase_optimize_hardware (FIFOs sized; ZynqBuild generates the IP).

    python run_frontend.py --bench vgg10-radioml --board U55C --clk 10 --out <dir> [--target-fps F]
    python run_frontend.py --bench bnn-pynq:cnv-w1a1 --board ZCU104 --clk 10 --out <dir>
    python run_frontend.py --bench resnet50 --config-board V80 --board U55C --pe-div 2 --out <dir>

The result model: <out>/frontend/intermediate_models/step_set_fifo_depths.onnx (the step's
output) -- the same path the earlier per-model runners produced.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark")


def load_test(bench):
    """(test module, model file) of a benchmark ("bnn-pynq:<model>" for the BNN family)."""
    sub, _, which = bench.partition(":")
    d = os.path.join(BENCH, sub)
    sys.path.insert(0, d)
    sys.path.insert(0, BENCH)
    test = [f for f in os.listdir(d) if f.startswith("test_build_") and f.endswith(".py")][0]
    spec = importlib.util.spec_from_file_location(test[:-3].replace("-", "_"), os.path.join(d, test))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if which:
        model = mod.get_model_file(which)
    else:
        model = mod.model_file
    return mod, model, which


def divided_folding(src, dst, div):
    cfg = json.load(open(src))
    n = 0
    for v in cfg.values():
        if isinstance(v, dict) and v.get("PE", 1) % div == 0 and v.get("PE", 1) >= div:
            v["PE"] //= div
            n += 1
    json.dump(cfg, open(dst, "w"), indent=2)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="vgg10-radioml | mobilenet_v1 | resnet50 | bnn-pynq:<model> | ...")
    ap.add_argument("--board", required=True)
    ap.add_argument("--config-board", default=None, help="board whose folding config is used (default --board)")
    ap.add_argument("--clk", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--no-relax", action="store_true")
    ap.add_argument("--pe-div", type=int, default=1)
    ap.add_argument("--mvau-ram", default=None,
                    help="ram_style for the MVAUs whose config says auto/block (e.g. ultra: ResNet50's V80 "
                    "config needs ~2000 BRAM36 on the U55C, which has 2016 but 960 URAMs)")
    ap.add_argument("--fifo", default="heuristic_analytical",
                    help="auto FIFO sizing strategy (heuristic_analytical | heuristic_rtlsim | largefifo_rtlsim | none)")
    args = ap.parse_args()
    mod, model, which = load_test(args.bench)
    front = os.path.join(args.out, "frontend")
    os.makedirs(front, exist_ok=True)
    cboard = args.config_board or args.board
    over = dict(
        synth_clk_period_ns=args.clk,
        stop_step="phase_optimize_hardware",
        # (a non-estimate output, so that the FIFOs are sized; the build stops before it)
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS, build_cfg.DataflowOutputType.BITFILE],
        verify_steps=None,
        verify_input_npy=None,
        verify_expected_output_npy=None,
        enable_build_pdb_debug=False,
    )
    if args.fifo == "none":
        over["auto_fifo_depths"] = False
    else:
        over["auto_fifo_depths"] = True
        over["auto_fifo_strategy"] = args.fifo
    cfg = mod.configure_build(cboard, which, front, **over) if which else mod.configure_build(cboard, front, **over)
    cfg.board = args.board
    if args.pe_div > 1:
        dst = os.path.join(args.out, "folding_pe_div%d.json" % args.pe_div)
        print("PE divided by %d in %d layers" % (args.pe_div, divided_folding(cfg.folding_config_file, dst, args.pe_div)))
        cfg.folding_config_file = dst
    if args.mvau_ram:
        fc = json.load(open(cfg.folding_config_file))
        n = 0
        for k, v in fc.items():
            if k.startswith("MVAU") and isinstance(v, dict) and v.get("ram_style", "auto") in ("auto", "block"):
                v["ram_style"] = args.mvau_ram
                n += 1
        dst = os.path.join(args.out, "folding_mvau_%s.json" % args.mvau_ram)
        json.dump(fc, open(dst, "w"), indent=2)
        cfg.folding_config_file = dst
        print("MVAU ram_style %s in %d layers" % (args.mvau_ram, n))
    if args.target_fps:
        from scaling_folding import scaling_kwargs

        for k, v in scaling_kwargs(cfg.folding_config_file, args.target_fps, args.out, not args.no_relax).items():
            setattr(cfg, k, v)
    t0 = time.time()
    build.build_dataflow_cfg(model, cfg)
    im = os.path.join(front, "intermediate_models")
    if not os.path.isfile(os.path.join(im, "step_set_fifo_depths.onnx")):
        # (FIFO sizing skipped or named differently: the phase's output)
        last = max((os.path.join(im, f) for f in os.listdir(im) if f.endswith(".onnx")), key=os.path.getmtime)
        shutil.copy(last, os.path.join(im, "step_set_fifo_depths.onnx"))
    res = {"bench": args.bench, "board": args.board, "config_board": cboard, "clk": args.clk,
           "target_fps": args.target_fps, "pe_div": args.pe_div, "total_s": time.time() - t0}
    json.dump(res, open(os.path.join(args.out, "frontend.json"), "w"), indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
