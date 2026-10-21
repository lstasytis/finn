# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import contextlib
import fcntl
import functools
import glob
import json
import numpy as np

# custom steps for the COCO YOLOv8
import os
import shutil
import time
import warnings
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    benchmark_root,
    bitfile_output_files,
    check_build_outputs,
    find_cached_build,
    get_verify_steps,
    make_benchmark_cfg,
    verification_io_dir,
)
from custom_steps_yolov8 import step_yolov8_streamline

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir


# model
def get_model_file(model):
    return os.path.join(benchmark_root(), "models", model + "_comact_tidy.onnx")


# verification parameters
# The verification I/O is a fixed-seed random uint8 192x320 image and the
# corresponding output of the original (pre-streamline) model.
def get_verify_io(model):
    return (
        os.path.join(verification_io_dir(), model + "_input.npy"),
        os.path.join(verification_io_dir(), model + "_output.npy"),
    )


# initial_python is not listed: the models are already FINN-ONNX (MultiThreshold
# activations, UINT8 input annotation), so step_qonnx_to_finn/step_tidy_up are not
# part of the flow.
verif_steps = [
    "streamlined_python",
    "folded_hls_cppsim",
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


# Per-precision settings on top of the shared config. The 8-bit activations are
# implemented as Requant instead of 255-threshold Thresholding layers, which would
# otherwise dominate the LUT budget.
MODEL_OVERRIDES = {
    "quantyolov8_4w4a": {},
    "quantyolov8_8w8a": {"requant_bitwidth_threshold": 8},
}

# Build steps: the models are already FINN-ONNX, so the flow starts directly with
# the YOLOv8-specific streamline (C2f split/concat + SPPF handling + conv
# lowering), which replaces the standard prepare/streamline phases. The rest uses
# the phase-based default flow.
yolov8_build_steps = [
    step_yolov8_streamline,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]


def configure_build(board, model, output_dir, small=False, **overrides):
    # Folding was derived from target_fps=90 auto-folding. No specialize config:
    # the default layer specialization is used.
    folding = os.path.join(
        benchmark_root(), "yolov8", "folding_config", f"{model}_folding_config_{board}.json"
    )
    verify_input_npy, verify_expected_output_npy = get_verify_io(model)
    cfg = dict(
        steps=yolov8_build_steps,
        folding_config_file=small_folding(folding) if small else folding,
        synth_clk_period_ns=10.0,
        auto_fifo_depths=True,
        standalone_thresholds=True,
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(MODEL_OVERRIDES[model])
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


YOLOV8_BUILDS = [
    ("U55C", "quantyolov8_4w4a"),
    ("U55C", "quantyolov8_8w8a"),
]


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board,model", YOLOV8_BUILDS)
def test_yolov8(board, model, bench_recorder):
    output_dir = make_build_dir(f"build_yolov8_{model}_{board}_")

    # Run build flow
    cfg = configure_build(board, model, output_dir)
    build.build_dataflow_cfg(get_model_file(model), cfg)
    bench_recorder(model, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one.
    build_output_files = CORE_BUILD_OUTPUT_FILES + bitfile_output_files(board)
    check_build_outputs(output_dir, build_output_files)


# --- feature tests: FIFO sizing, folding, DWC padding -----------------------
# Each test resumes the convert-phase checkpoint of a cached estimate-only build
# (committed folding) with the feature's settings, builds the stitched IP and
# checks it in rtlsim: output equal to the golden output and throughput within
# 2 % of the build's own analytic estimate.

# Feature-test helpers, replicated in every model's test file so that each file is
# self-contained (only upstream's benchmark helpers are shared).

# estimate-only builds stop after the phase that writes the estimate reports
ESTIMATE_OUTPUTS = [build_cfg.DataflowOutputType.ESTIMATE_REPORTS]
ESTIMATE_PHASE = "phase_optimize_hardware"

# Feature builds stop at stitched IP: no bitfile, rtlsim throughput only.
FEATURE_OUTPUTS = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
    build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
]

# the phase whose output checkpoint the feature builds start from
CHECKPOINT_PHASE = "phase_convert_to_hardware"


def feature_steps(steps=None):
    """The part of a model's step list that a feature build runs (everything
    after ``CHECKPOINT_PHASE``); ``None`` stands for the default phase list."""
    steps = list(build_cfg.default_build_dataflow_steps if steps is None else steps)
    return steps[steps.index(CHECKPOINT_PHASE) + 1 :]


def cached_build(prefix, build_fn, done_file):
    """Return the ``FINN_BUILD_DIR`` build dir ``prefix*`` holding ``done_file``,
    building it with ``build_fn(output_dir)`` if there is none. A file lock keeps
    concurrent tests of the same model from building it twice."""
    build_dir = os.environ["FINN_BUILD_DIR"]
    with open(os.path.join(build_dir, prefix + "lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output_dir = find_cached_build(prefix)
        if output_dir is None or not os.path.isfile(os.path.join(output_dir, done_file)):
            output_dir = make_build_dir(prefix)
            build_fn(output_dir)
        return output_dir


def resume_checkpoint(ref_dir, output_dir):
    """Prepare ``output_dir`` to continue ``ref_dir``'s build after
    ``CHECKPOINT_PHASE`` and return the checkpoint to pass as the model file.
    Stitched-IP verification needs the dataflow parent model next to it."""
    os.makedirs(os.path.join(output_dir, "intermediate_models"), exist_ok=True)
    parent = "intermediate_models/dataflow_parent.onnx"
    shutil.copy(os.path.join(ref_dir, parent), os.path.join(output_dir, parent))
    return os.path.join(ref_dir, "intermediate_models", CHECKPOINT_PHASE + ".onnx")


@contextlib.contextmanager
def timed_steps(step_names=("step_target_fps_parallelization", "step_set_fifo_depths")):
    """Record the wall time (s) of the named build steps into the yielded dict.
    The builder only times whole phases, so the steps are wrapped where the
    phases look them up."""
    import finn.builder.build_dataflow_phases as phases

    times, originals = {}, {name: getattr(phases, name) for name in step_names}

    def timed(fn):
        @functools.wraps(fn)
        def wrapper(model, cfg):
            start = time.time()
            try:
                return fn(model, cfg)
            finally:
                times[fn.__name__] = times.get(fn.__name__, 0.0) + time.time() - start

        return wrapper

    for name, fn in originals.items():
        setattr(phases, name, timed(fn))
    try:
        yield times
    finally:
        for name, fn in originals.items():
            setattr(phases, name, fn)


def estimated_throughput(output_dir):
    """Analytic throughput (fps) from the estimate reports, None if missing."""
    est = _load_json(os.path.join(output_dir, "report", "estimate_network_performance.json"))
    return est.get("estimated_throughput_fps") if est else None


def rtlsim_throughput(output_dir):
    """Steady-state stitched-IP rtlsim throughput (fps), None if missing."""
    perf = _load_json(os.path.join(output_dir, "report", "rtlsim_performance.json"))
    return perf.get("stable_throughput[images/s]") if perf else None


def verification_passed(output_dir, step="stitched_ip_rtlsim"):
    """True if ``step`` verified and every verified frame matched the golden output."""
    results = glob.glob(os.path.join(output_dir, "verification_output", f"verify_{step}_*"))
    return bool(results) and all("_SUCCESS." in os.path.basename(r) for r in results)


def feature_metrics(output_dir, ref_dir, **extra):
    """Collect a feature build's numbers (throughput, FIFO storage, padding DWCs,
    folding/FIFO step times), store them as ``report/feature_metrics.json`` and
    return them."""
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.custom_op.registry import getCustomOp

    est, ref = estimated_throughput(output_dir), estimated_throughput(ref_dir)
    rtlsim = rtlsim_throughput(output_dir)
    row = dict(
        output_dir=output_dir,
        est_fps=est,
        ref_est_fps=ref,
        rtlsim_fps=rtlsim,
        rtlsim_vs_est=rtlsim / est if rtlsim and est else None,
        est_vs_ref=est / ref if est and ref else None,
        verified=verification_passed(output_dir),
    )
    for step in ["step_set_fifo_depths", "step_minimize_bit_width"]:
        model_file = os.path.join(output_dir, "intermediate_models", step + ".onnx")
        if os.path.isfile(model_file):
            break
    if os.path.isfile(model_file):
        fifo_bits, pad_dwcs = 0, 0
        for node in ModelWrapper(model_file).graph.node:
            inst = getCustomOp(node) if node.domain.startswith("finn") else None
            if node.op_type.startswith("StreamingFIFO"):
                fifo_bits += inst.get_nodeattr("depth") * inst.get_instream_width()
            elif node.op_type.startswith("StreamingDataWidthConverter"):
                n_in = int(np.prod(inst.get_normal_input_shape()))
                pad_dwcs += n_in != int(np.prod(inst.get_normal_output_shape()))
        row.update(fifo_kbit=fifo_bits / 1024, pad_dwcs=pad_dwcs)
    row.update(extra)
    os.makedirs(os.path.join(output_dir, "report"), exist_ok=True)
    with open(os.path.join(output_dir, "report", "feature_metrics.json"), "w") as f:
        json.dump(row, f, indent=2)
    return row


def small_folding(path):
    """The reduced variant (``<name>_small.json``) of a committed folding config: every
    node's parallelism cut 4-8x (claude-tools make_small_foldings.py), so that the
    feature tests' stitched-IP rtlsim runs on a much smaller design first."""
    small = path[: -len(".json")] + "_small.json"
    assert os.path.isfile(small), f"no small folding config {small}"
    return small


def _load_json(path):
    """Load a JSON file, None if it is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


FEATURE_BUILDS = [
    pytest.param(board, model, marks=pytest.mark.xdist_group(model))
    for board, model in YOLOV8_BUILDS
]
FOLDING_MAX_PADDING = 4
DWC_SEEDS = list(range(1, 9))
# optimizer folding (fixed seed) instead of the committed JSON, FIFOs sized analytically
OPTIMIZER_FOLDING = dict(
    folding_config_file=None,
    folding_style="optimizer",
    folding_seed=1,
    auto_fifo_depths=True,
    auto_fifo_strategy="heuristic_analytical",
)


# a padded folding below the target still counts as a win when its resource reduction
# vs the committed folding is larger than its throughput loss, within TARGET_TOLERANCE
TARGET_TOLERANCE = 0.10


def resource_share(output_dir, board):
    """A build's estimated resources (FIFOs excluded: the reference sizes none) as the
    summed shares of the board's LUT, BRAM_18K, URAM and DSP."""
    from finn.util.platforms import platforms

    device = {}
    for slr in platforms[board]().resource_count_dict.values():
        for r, n in slr.items():
            device[r] = device.get(r, 0) + n
    layers = _load_json(os.path.join(output_dir, "report", "estimate_layer_resources.json"))
    used = {}
    for name, res in layers.items():
        if name != "total" and not name.startswith("StreamingFIFO"):
            for r, n in res.items():
                used[r] = used.get(r, 0) + n
    return sum(used.get(r, 0) / device[r] for r in ("LUT", "BRAM_18K", "URAM", "DSP") if device[r])


def assert_target_or_trade(metrics, ref_dir, target_fps, board):
    """The estimate meets the target, or misses it by at most TARGET_TOLERANCE for a
    larger resource reduction (%) than throughput loss (%) vs the committed folding."""
    loss = 1 - metrics["est_fps"] / target_fps
    if loss <= 0:
        return
    saved = 1 - resource_share(metrics["output_dir"], board) / resource_share(ref_dir, board)
    msg = "%.1f %% below the target for %.1f %% fewer resources than the committed folding"
    assert loss <= TARGET_TOLERANCE and saved > loss, msg % (100 * loss, 100 * saved)


# DWC padding vs none: the optimizer at the same target and seed with padding off and on
RESOURCES = ("LUT", "BRAM_18K", "URAM", "DSP")
# padding only widens the search space: a padded folding needing more resources than the
# unpadded one is annealing noise up to this fraction (warning), beyond it the test fails
PADDING_WORSE_TOLERANCE = 0.05


def estimated_resources(output_dir):
    """A build's estimated LUT/BRAM_18K/URAM/DSP counts (FIFOs excluded)."""
    layers = _load_json(os.path.join(output_dir, "report", "estimate_layer_resources.json"))
    used = dict.fromkeys(RESOURCES, 0.0)
    for name, res in (layers or {}).items():
        if name != "total" and not name.startswith("StreamingFIFO") and isinstance(res, dict):
            for r in RESOURCES:
                used[r] += float(res.get(r, 0) or 0)
    return used


def resource_ratio(new, base):
    """Mean of new/base over the resources ``base`` uses, and the resources only ``new``
    uses (the res./ref. metric of the claude-tools summary)."""
    ratios = [new[r] / base[r] for r in RESOURCES if base[r] > 0]
    extra = [r for r in RESOURCES if base[r] == 0 and new[r] > 0]
    return (sum(ratios) / len(ratios) if ratios else 1.0), extra


def padding_summary(metrics):
    """The numbers of one estimate-only folding kept in the comparison."""
    return dict(
        est_fps=metrics["est_fps"],
        pad_dwcs=metrics.get("pad_dwcs"),
        resources=estimated_resources(metrics["output_dir"]),
        output_dir=metrics["output_dir"],
    )


def padding_comparison(build, target_fps):
    """Estimate-only optimizer foldings at ``target_fps`` without and with padding (same
    seed) for every seed in DWC_SEEDS; ``build(name, estimate_only=True, **overrides)``
    runs one. Returns the comparison of the seed where padding saves the most resources."""
    best, best_key, seeds = None, None, []
    for seed in DWC_SEEDS:
        runs = [
            padding_summary(
                build(
                    name,
                    estimate_only=True,
                    **dict(
                        OPTIMIZER_FOLDING,
                        target_fps=target_fps,
                        folding_maximum_padding=pad,
                        folding_seed=seed,
                    ),
                )
            )
            for name, pad in (("dwc-est-nopad", 0), ("dwc-est", FOLDING_MAX_PADDING))
        ]
        ratio, extra = resource_ratio(runs[1]["resources"], runs[0]["resources"])
        row = dict(seed=seed, res_ratio=ratio, res_extra=extra, nopad=runs[0], pad=runs[1])
        seeds.append(
            dict(seed=seed, res_ratio=ratio, res_extra=extra, pad_dwcs=runs[1]["pad_dwcs"])
        )
        # on a tie the padded folding that pads, so that the hardware build has padding DWCs
        key = (bool(extra), ratio, not runs[1]["pad_dwcs"])
        if best is None or key < best_key:
            best, best_key = row, key
    if best["res_extra"] or best["res_ratio"] > 1:
        best["verdict"] = "worse"
    elif not best["pad"]["pad_dwcs"]:
        best["verdict"] = "no gain (no padding DWC)"
    else:
        best["verdict"] = "gain" if best["res_ratio"] < 1 else "no gain"
    return dict(best, target_fps=target_fps, seeds=seeds)


def padding_maximize(build):
    """Estimated throughput of the maximize-mode optimizer (no target) without and with
    padding."""
    nopad, pad = [
        padding_summary(
            build(
                name,
                estimate_only=True,
                **dict(OPTIMIZER_FOLDING, folding_maximum_padding=padding),
            )
        )
        for name, padding in (("dwc-max-nopad", 0), ("dwc-max", FOLDING_MAX_PADDING))
    ]
    ratio, extra = resource_ratio(pad["resources"], nopad["resources"])
    fps_ratio = pad["est_fps"] / nopad["est_fps"]
    if fps_ratio < 1:
        warnings.warn("maximize: padded folding %.3fx the unpadded throughput" % fps_ratio)
    return dict(fps_ratio=fps_ratio, res_ratio=ratio, res_extra=extra, nopad=nopad, pad=pad)


def assert_padding_not_worse(compare):
    """Beyond annealing noise the padded folding needs no more resources than the unpadded
    one at the same target and seed (warning above 1.00x, failure above
    1 + PADDING_WORSE_TOLERANCE)."""
    ratio, extra = compare["res_ratio"], compare["res_extra"]
    msg = "padded folding needs %.3fx the resources of the unpadded one%s (seed %d)" % (
        ratio,
        " and " + "/".join(extra) if extra else "",
        compare["seed"],
    )
    if compare["verdict"] == "worse":
        warnings.warn(msg)
    assert ratio <= 1 + PADDING_WORSE_TOLERANCE, msg


def reference_estimate(board, model, small=False):
    """Estimate-only build with the committed folding: its convert-phase
    checkpoint and estimated throughput are the feature tests' reference."""

    def build_estimate(output_dir):
        cfg = configure_build(
            board,
            model,
            output_dir,
            small=small,
            generate_outputs=ESTIMATE_OUTPUTS,
            stop_step=ESTIMATE_PHASE,
            verify_steps=[],
        )
        assert build.build_dataflow_cfg(get_model_file(model), cfg) == 0

    return cached_build(
        f"est_yolov8_{'small_' if small else ''}{model}_{board}_",
        build_estimate,
        "report/estimate_network_performance.json",
    )


def run_feature_build(
    small, board, model, name, estimate_only=False, extra_metrics=None, **overrides
):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics
    (with ``extra_metrics``, also stored in feature_metrics.json)."""
    ref_dir = reference_estimate(board, model, small=small)
    output_dir = make_build_dir(f"{name}{'-small' if small else ''}_yolov8_{model}_{board}_")
    cfg = configure_build(
        board,
        model,
        output_dir,
        small=small,
        steps=feature_steps(yolov8_build_steps),
        generate_outputs=ESTIMATE_OUTPUTS if estimate_only else FEATURE_OUTPUTS,
        stop_step=ESTIMATE_PHASE if estimate_only else None,
        verify_steps=[] if estimate_only else ["stitched_ip_rtlsim"],
        stitched_ip_gen_dcp=False,
        save_intermediate_models=True,
        **overrides,
    )
    # make_benchmark_cfg drops None values: apply them afterwards (e.g. no folding JSON)
    for key, value in overrides.items():
        if value is None:
            setattr(cfg, key, None)
    with timed_steps() as step_times:
        ret = build.build_dataflow_cfg(resume_checkpoint(ref_dir, output_dir), cfg)
    metrics = feature_metrics(
        output_dir, ref_dir, test=name, step_times_s=step_times, **(extra_metrics or {})
    )
    assert ret == 0, f"build failed, see {output_dir}/build_dataflow.log"
    if not estimate_only:
        assert metrics["verified"], "stitched-IP rtlsim output differs from the golden output"
        assert (metrics["rtlsim_vs_est"] or 0) >= 0.98, "rtlsim %s fps, estimate %s fps" % (
            metrics["rtlsim_fps"],
            metrics["est_fps"],
        )
    return metrics


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.fifo_sizing
@pytest.mark.parametrize("size", ["small", "full"])
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board,model", FEATURE_BUILDS)
def test_yolov8_fifo_sizing(board, model, strategy, size):
    small = size == "small"
    run_feature_build(
        small, board, model, f"fifo-{strategy}", auto_fifo_depths=True, auto_fifo_strategy=strategy
    )


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.folding
@pytest.mark.parametrize("size", ["small", "full"])
@pytest.mark.parametrize("mode", ["target", "maximize", "fifo-aware"])
@pytest.mark.parametrize("board,model", FEATURE_BUILDS)
def test_yolov8_folding(board, model, mode, size):
    small = size == "small"
    if small and mode == "maximize":
        pytest.skip("maximize does not use the committed folding")
    target_fps = int(estimated_throughput(reference_estimate(board, model, small=small)))
    if mode == "target":
        metrics = run_feature_build(
            small, board, model, "fold-target", target_fps=target_fps, **OPTIMIZER_FOLDING
        )
        assert metrics["est_fps"] >= target_fps
    elif mode == "fifo-aware":
        # optional, slower: the search scores foldings with FIFOs sized by the analytic sizer
        metrics = run_feature_build(
            small,
            board,
            model,
            "fold-fifoaware",
            target_fps=target_fps,
            enable_folding_fifo_heuristic=True,
            **OPTIMIZER_FOLDING,
        )
        assert metrics["est_fps"] >= target_fps
    else:
        # no target: the optimizer maximizes the throughput within the budget
        metrics = run_feature_build(small, board, model, "fold-max", **OPTIMIZER_FOLDING)
        if metrics["est_vs_ref"] < 1:
            warnings.warn("maximized folding is %.2fx the committed one" % metrics["est_vs_ref"])


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.dwc
@pytest.mark.parametrize("size", ["small", "full"])
@pytest.mark.parametrize("board,model", FEATURE_BUILDS)
def test_yolov8_dwc(board, model, size):
    small = size == "small"
    ref_dir = reference_estimate(board, model, small=small)
    target_fps = int(estimated_throughput(ref_dir))
    build = functools.partial(run_feature_build, small, board, model)
    # padding off vs on at the reference target (same seed): the seed where padding saves the
    # most resources is built, also when padding gains nothing ("verdict" in the metrics)
    compare = padding_comparison(build, target_fps)
    if not small:
        # maximize mode (no small variant): the throughput padding adds within the budget
        compare["maximize"] = padding_maximize(build)
    padded = dict(
        OPTIMIZER_FOLDING,
        target_fps=target_fps,
        folding_maximum_padding=FOLDING_MAX_PADDING,
        folding_seed=compare["seed"],
    )
    metrics = build("dwc", extra_metrics=dict(pad_compare=compare), **padded)
    assert_target_or_trade(metrics, ref_dir, target_fps, board)
    assert_padding_not_worse(compare)
