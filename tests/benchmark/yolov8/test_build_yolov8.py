# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

# custom steps for the COCO YOLOv8
import os
import warnings
from benchmark_helpers import (
    CORE_BUILD_OUTPUT_FILES,
    ESTIMATE_OUTPUTS,
    ESTIMATE_PHASE,
    FEATURE_OUTPUTS,
    benchmark_root,
    bitfile_output_files,
    cached_build,
    check_build_outputs,
    estimated_throughput,
    feature_metrics,
    feature_steps,
    get_verify_steps,
    make_benchmark_cfg,
    resume_checkpoint,
    small_folding,
    timed_steps,
    verification_io_dir,
)
from custom_steps_yolov8 import step_yolov8_streamline

import finn.builder.build_dataflow as build
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


def run_feature_build(small, board, model, name, estimate_only=False, **overrides):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics."""
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
    metrics = feature_metrics(output_dir, ref_dir, test=name, step_times_s=step_times)
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
@pytest.mark.parametrize("mode", ["target", "maximize"])
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
    target_fps = int(estimated_throughput(reference_estimate(board, model, small=small)))
    # the folding search is cheap: find a seed whose padded folding pads (padding wins
    # only on some search paths), skip the hardware build if none does
    for seed in DWC_SEEDS:
        padded = dict(
            OPTIMIZER_FOLDING,
            target_fps=target_fps,
            folding_maximum_padding=FOLDING_MAX_PADDING,
            folding_seed=seed,
        )
        if run_feature_build(small, board, model, "dwc-est", estimate_only=True, **padded)[
            "pad_dwcs"
        ]:
            break
    else:
        pytest.skip("no padding needed: no padded folding inserts a padding DWC")
    metrics = run_feature_build(small, board, model, "dwc", **padded)
    assert metrics["est_fps"] >= target_fps
