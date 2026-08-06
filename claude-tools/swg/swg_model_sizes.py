# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Model-level guard: total FIFO KB of a finn-examples build, tree models only.

The per-config harness (``swg_tav.py``) says how far a candidate SWG model has
moved; this one says what that costs. It runs a build truncated at
``step_set_fifo_depths`` -- no ipgen, no synthesis, no rtlsim -- and reports the
FIFO total, per model and per FIFO.

The expensive part of the run is everything *before* sizing, and none of it
depends on the tree model, so it is cached: the first run of a model writes
``<builddir>/intermediate_models/step_hw_ipgen.onnx`` (or the last step before
sizing) and every later run starts from it and re-runs sizing alone. That turns
an iteration on the SWG model from tens of minutes into under a minute.

    python claude-tools/swg/swg_model_sizes.py build   --model mobilenet_v1
    python claude-tools/swg/swg_model_sizes.py size    --model mobilenet_v1 -o base.json
    python claude-tools/swg/swg_model_sizes.py compare -a base.json -b candidate.json

A candidate whose total is **lower** than the baseline is a failure, not a win:
the baseline is the depth the board was validated at, so less is undersizing.
``compare`` therefore reports the signed delta and fails on either direction
outside its budget.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault(
    "FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

MODELS = {
    "mobilenet_v1": dict(
        onnx="tests/benchmark/models/mobilenetv1-w4a4_pre_post_tidy_opset-11.onnx",
        folding="tests/benchmark/mobilenet_v1/folding_config/mobilenet_folding_config_ZCU104.json",
        specialize=(
            "tests/benchmark/mobilenet_v1/specialize_layers_config/"
            "mobilenet_specialize_layers_ZCU104.json"
        ),
        custom_steps="tests/benchmark/mobilenet_v1",
        board="ZCU104",
        clk=5.4,
        standalone_thresholds=True,
    ),
    # The second sliding-window-heavy model that actually builds. resnet50 is
    # the one this workstream would rather use -- more CIG nodes, and the
    # branch-sizing path -- but it does not get as far as FIFO sizing on this
    # tree: step_create_dataflow_partition dies with "cycle-free graph
    # violated: partition depends on itself", a known pre-existing blocker
    # unrelated to tree models. cnv-w2a2 has 8 CIG nodes and builds in minutes.
    "cnv-w2a2": dict(
        onnx="tests/benchmark/models/cnv-w2a2.onnx",
        folding="tests/benchmark/bnn-pynq/folding_config/cnv-w2a2_folding_config.json",
        specialize=(
            "tests/benchmark/bnn-pynq/specialize_layers_config/cnv-w2a2_specialize_layers.json"
        ),
        custom_steps="tests/benchmark/bnn-pynq",
        board="Pynq-Z1",
    ),
    "resnet50": dict(
        onnx="tests/benchmark/models/resnet50_w1a2_exported.onnx",
        folding="tests/benchmark/resnet50/folding_config/resnet50_folding_config_U250.json",
        specialize=(
            "tests/benchmark/resnet50/specialize_layers_config/"
            "resnet50_specialize_layers_U250.json"
        ),
        custom_steps="tests/benchmark/resnet50",
        board="U250",
    ),
}


def build_dir(model):
    return os.path.join(os.environ["FINN_BUILD_DIR"], "swg_fifo_%s" % model)


def _cfg(model, steps, output_dir):
    """A build config truncated at FIFO sizing, analytic tree-model strategy.

    The step list is assembled from the model's own benchmark test rather than
    the default, because both mobilenet and resnet50 need custom streamlining
    steps. Truncating by name is the documented way to cut a dev-phase build
    short -- see claude-tools/README.md.
    """
    import finn.builder.build_dataflow_config as build_cfg

    spec = MODELS[model]
    sys.path.insert(0, os.path.abspath(spec["custom_steps"]))
    return build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        steps=steps,
        folding_config_file=spec["folding"],
        specialize_layers_config_file=spec["specialize"],
        synth_clk_period_ns=spec.get("clk", 5.0),
        # Not cosmetic: with standalone_thresholds off, MatMul+MultiThreshold
        # fuse into the MVAU and the graph stops being the one the folding
        # config was written against -- on ZCU104 that shows up as
        # "MH divisable by PE is violated" and a DWC with a non-integer ratio.
        # Each board's benchmark test picks this, so mirror it per model.
        standalone_thresholds=spec.get("standalone_thresholds", False),
        board=spec["board"],
        shell_flow_type=(
            build_cfg.ShellFlowType.VITIS_ALVEO
            if spec["board"] == "U250"
            else build_cfg.ShellFlowType.VIVADO_ZYNQ
        ),
        auto_fifo_depths=True,
        # tree models, no rtlsim; chained_tav is the sizer default
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod.HEURISTIC_ANALYTICAL,
        heuristic_fifo_sizing_method=build_cfg.HeuristicFifoSizingMethod.CHAINED_TAV,
        generate_outputs=[],
        enable_build_pdb_debug=False,
        verbose=True,
    )


def steps_for(model, upto, with_ipgen=False):
    """The model's own benchmark step list, truncated *before* ``upto``.

    Taken from the benchmark test rather than the default list: both models
    need custom streamlining. ``step_hw_ipgen`` is dropped unless asked for --
    analytical sizing reads tree models, not IP, and ipgen is the single most
    expensive step before sizing. Put it back with ``--with-ipgen`` if a node
    without a tree model has to be synthesised just-in-time.
    """
    sys.path.insert(0, os.path.abspath(MODELS[model]["custom_steps"]))
    if model == "mobilenet_v1":
        from test_build_mobilenet_v1 import select_build_steps

        steps = select_build_steps(MODELS[model]["board"])
    elif model == "resnet50":
        from test_build_resnet50 import resnet50_build_steps as steps
    else:
        # bnn-pynq needs no custom steps, but the default list is the six
        # *phases* and phase_optimize_hardware bundles sizing in with folding,
        # so it cannot be truncated between the two. Name the steps instead.
        steps = [
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
    if not with_ipgen:
        steps = [s for s in steps if s != "step_hw_ipgen"]
    idx = steps.index(upto)
    return steps[:idx], steps[: idx + 1]


def cmd_build(args):
    """Run everything up to (not including) FIFO sizing and leave the checkpoint."""
    import finn.builder.build_dataflow as build

    out = args.output_dir or build_dir(args.model)
    pre, _ = steps_for(args.model, "step_set_fifo_depths", args.with_ipgen)
    cfg = _cfg(args.model, pre, out)
    t0 = time.time()
    rc = build.build_dataflow_cfg(MODELS[args.model]["onnx"], cfg)
    print("build_dataflow_cfg -> %s in %.0fs (checkpoint in %s)" % (rc, time.time() - t0, out))
    return 0 if rc != -1 else 1


def cmd_size(args):
    """Re-run FIFO sizing alone on the cached checkpoint and report the KB.

    ``SWG_CAPPROBE=<dir>`` imports ``capprobe`` from that directory first, which
    collects the sizer's per-edge trace and dumps the rows for one consumer at
    exit. That is how the throttled cap's terms were measured.
    """
    if os.environ.get("SWG_CAPPROBE"):
        sys.path.insert(0, os.environ["SWG_CAPPROBE"])
        import capprobe  # noqa: F401

    from qonnx.core.modelwrapper import ModelWrapper

    import finn.builder.build_dataflow as build

    out = args.output_dir or build_dir(args.model)
    pre, _ = steps_for(args.model, "step_set_fifo_depths", args.with_ipgen)
    last = getattr(pre[-1], "__name__", pre[-1])
    ckpt = os.path.join(out, "intermediate_models", "%s.onnx" % last)
    if not os.path.exists(ckpt):
        print("no checkpoint at %s -- run `build` first" % ckpt)
        return 1
    cfg = _cfg(args.model, ["step_set_fifo_depths"], out)
    t0 = time.time()
    rc = build.build_dataflow_cfg(ckpt, cfg)
    if rc == -1:
        print("sizing failed")
        return 1
    sized = os.path.join(out, "intermediate_models", "step_set_fifo_depths.onnx")
    report = fifo_report(ModelWrapper(sized))
    report["model"] = args.model
    report["seconds"] = round(time.time() - t0, 1)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    print(
        "%s: %d FIFOs, %.1f kB total (%.0fs)"
        % (args.model, len(report["fifos"]), report["total_kb"], report["seconds"])
    )
    for name, d in sorted(report["fifos"].items(), key=lambda kv: -kv[1]["kb"])[: args.top]:
        print("  %-48s depth=%-7d width=%-4d %8.2f kB" % (name, d["depth"], d["width"], d["kb"]))
    return 0


def fifo_report(model):
    """Per-FIFO depth/width/kB of a sized model, and the total.

    Width is the **folded stream width** -- what the AXI stream actually
    carries, and what the FIFO is that many entries deep of. The normal shape's
    last dimension is the unfolded tensor channel count, which for a windowed
    tensor is `k*k*C` and overstates the storage by the folding factor: on
    mobilenet_v1 that is the difference between 3686 kB and 88 kB.
    """
    from qonnx.custom_op.registry import getCustomOp

    fifos, total_bits = {}, 0
    for node in model.graph.node:
        if node.op_type != "StreamingFIFO_rtl":
            continue
        inst = getCustomOp(node)
        depth = inst.get_nodeattr("depth")
        width = inst.get_outstream_width()
        bits = depth * width
        total_bits += bits
        prod = model.find_producer(node.input[0])
        cons = model.find_consumer(node.output[0])
        fifos[node.name] = dict(
            depth=int(depth),
            width=int(width),
            kb=bits / 8 / 1024,
            edge="%s -> %s"
            % (
                prod.op_type if prod is not None else "-",
                cons.op_type if cons is not None else "-",
            ),
        )
    return dict(total_kb=total_bits / 8 / 1024, total_bits=int(total_bits), fifos=fifos)


def cmd_compare(args):
    a, b = json.load(open(args.a)), json.load(open(args.b))
    da = a["total_kb"]
    db = b["total_kb"]
    frac = (db - da) / da if da else 0.0
    print("%s: %.1f kB -> %.1f kB  (%+.2f%%)" % (a.get("model", "?"), da, db, 100 * frac))
    moved = []
    for name, fa in a["fifos"].items():
        fb = b["fifos"].get(name)
        if fb is None:
            moved.append((name, fa["depth"], None))
        elif fb["depth"] != fa["depth"]:
            moved.append((name, fa["depth"], fb["depth"]))
    for name, x, y in sorted(moved, key=lambda t: -(abs((t[2] or 0) - t[1])))[: args.top]:
        print("  %-48s %s -> %s" % (name, x, y))
    if frac < -args.tol:
        print("FAIL: total dropped by more than %.0f%% -- undersizing risk" % (100 * args.tol))
        return 1
    if frac > args.grow:
        print("FAIL: total grew by more than %.0f%%" % (100 * args.grow))
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "size"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, choices=sorted(MODELS))
        p.add_argument("--output-dir", default=None)
        p.add_argument("--with-ipgen", action="store_true")
        if name == "size":
            p.add_argument("-o", "--out", default=None)
            p.add_argument("--top", type=int, default=15)
        p.set_defaults(fn=cmd_build if name == "build" else cmd_size)
    c = sub.add_parser("compare")
    c.add_argument("-a", required=True, help="baseline json")
    c.add_argument("-b", required=True, help="candidate json")
    c.add_argument("--tol", type=float, default=0.0, help="allowed shrink fraction")
    c.add_argument("--grow", type=float, default=0.10, help="allowed growth fraction")
    c.add_argument("--top", type=int, default=15)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
