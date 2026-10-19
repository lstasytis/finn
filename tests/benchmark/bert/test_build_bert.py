# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file are derived from Brainsmith:
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
BERT (1-layer, int8) dataflow build benchmark.

This test reproduces the Brainsmith BERT blueprint flow as a plain FINN
``DataflowBuildConfig`` build. The blueprint's step pipeline is expressed
directly as the ``steps`` list below (mixing stock phases with the ported custom
steps in ``custom_steps_bert.py``), and the blueprint's ``finn_config`` block is
translated into ``DataflowBuildConfig`` arguments.
"""

import pytest

import os
import warnings
from benchmark_helpers import (
    ESTIMATE_OUTPUTS,
    ESTIMATE_PHASE,
    FEATURE_OUTPUTS,
    benchmark_root,
    cached_build,
    check_build_outputs,
    estimated_throughput,
    feature_metrics,
    feature_steps,
    get_verify_steps,
    make_benchmark_cfg,
    resume_checkpoint,
    timed_steps,
    verification_io_dir,
)
from custom_steps_bert import (
    step_bert_cleanup,
    step_bert_streamlining,
    step_remove_head,
)

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

# model
model_name = "l1_bert_quantized_int8"
model_file = os.path.join(benchmark_root(), "models", model_name + ".onnx")

# verification parameters
# The verification I/O is a fixed-seed uniform [0, 1000) input for the
# head-removed model (input = output of the first LayerNorm) and the
# corresponding output of the head-removed model before step_qonnx_to_finn.
verify_input_npy = os.path.join(verification_io_dir(), model_name + "_input.npy")
verify_expected_output_npy = os.path.join(verification_io_dir(), model_name + "_output.npy")

verif_steps = [
    "finn_onnx_python",
    "stitched_ip_rtlsim",
]

# Build steps: the embedding head (everything up to and including the first
# LayerNorm) is removed before the QONNX->FINN conversion, and the BERT-specific
# streamline (softmax Mul handling, thresholds kept per-tensor) replaces the
# standard streamline phase. The rest uses the phase-based default flow.
bert_build_steps = [
    step_bert_cleanup,
    step_remove_head,
    "phase_prepare_model",
    step_bert_streamlining,
    "phase_convert_to_hardware",
    "phase_optimize_hardware",
    "phase_build_hardware",
    "phase_generate_outputs",
]

# build output products
# rtlsim performance measurement is skipped for BERT to keep the build time down.
# TEMP(stitched-ip-only): the VCK190 bitfile flow needs the PYNQ golden reference
# design (FINN_VCK190_GOLDEN=1 ./run-docker.sh), so for now we stop at stitched IP.
# The output products gate which steps actually execute, so leaving
# BITFILE/PYNQ_DRIVER/DEPLOYMENT_PACKAGE out skips synth/driver/deployment. Re-add
# them once the golden reference is in place.
build_outputs = [
    build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
    build_cfg.DataflowOutputType.STITCHED_IP,
]


def configure_build(board, output_dir, **overrides):
    # Folding is hand-tuned for the VCK190; no specialize config, the default
    # layer specialization is used.
    folding = os.path.join(
        benchmark_root(), "bert", "folding_config", f"bert_folding_config_{board}.json"
    )
    cfg = dict(
        steps=bert_build_steps,
        generate_outputs=build_outputs,
        folding_config_file=folding,
        synth_clk_period_ns=5.0,
        standalone_thresholds=True,
        auto_fifo_depths=True,
        fifosim_n_inferences=2,
        verify_steps=get_verify_steps(verif_steps),
        verify_input_npy=verify_input_npy,
        verify_expected_output_npy=verify_expected_output_npy,
    )
    cfg.update(overrides)
    return make_benchmark_cfg(board, output_dir, **cfg)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("board", ["VCK190"])
def test_bert(board, bench_recorder):
    output_dir = make_build_dir(f"build_bert_{board}_")

    # Run build flow
    cfg = configure_build(board, output_dir)
    build.build_dataflow_cfg(model_file, cfg)
    bench_recorder(model_name, board, output_dir)

    # Check that all expected output products are present, reporting every
    # missing artifact at once instead of aborting on the first one.
    # TEMP(stitched-ip-only): bitfile/driver/deployment artifacts are omitted
    # while the VCK190 golden reference is unavailable; re-add
    # bitfile_output_files(board) and driver/driver.py once enabled.
    build_output_files = [
        "time_per_step.json",
        "final_hw_config.json",
        "template_specialize_layers_config.json",
        "stitched_ip/ip/component.xml",
        "report/estimate_layer_cycles.json",
        "report/estimate_layer_resources.json",
        "report/estimate_network_performance.json",
    ]
    check_build_outputs(output_dir, build_output_files)


# --- feature tests: FIFO sizing, folding, DWC padding -----------------------
# Each test resumes the convert-phase checkpoint of a cached estimate-only build
# (committed folding) with the feature's settings, builds the stitched IP and
# checks it in rtlsim: output equal to the golden output and throughput within
# 2 % of the build's own analytic estimate.

FEATURE_BUILDS = ["VCK190"]
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


def reference_estimate(board):
    """Estimate-only build with the committed folding: its convert-phase
    checkpoint and estimated throughput are the feature tests' reference."""

    def build_estimate(output_dir):
        cfg = configure_build(
            board,
            output_dir,
            generate_outputs=ESTIMATE_OUTPUTS,
            stop_step=ESTIMATE_PHASE,
            verify_steps=[],
        )
        assert build.build_dataflow_cfg(model_file, cfg) == 0

    return cached_build(
        f"est_bert_{board}_", build_estimate, "report/estimate_network_performance.json"
    )


def run_feature_build(board, name, estimate_only=False, **overrides):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics."""
    ref_dir = reference_estimate(board)
    output_dir = make_build_dir(f"{name}_bert_{board}_")
    cfg = configure_build(
        board,
        output_dir,
        steps=feature_steps(bert_build_steps),
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
@pytest.mark.xdist_group("bert")
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_bert_fifo_sizing(board, strategy):
    run_feature_build(board, f"fifo-{strategy}", auto_fifo_depths=True, auto_fifo_strategy=strategy)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.folding
@pytest.mark.xdist_group("bert")
@pytest.mark.parametrize("mode", ["target", "maximize"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_bert_folding(board, mode):
    target_fps = int(estimated_throughput(reference_estimate(board)))
    if mode == "target":
        metrics = run_feature_build(
            board, "fold-target", target_fps=target_fps, **OPTIMIZER_FOLDING
        )
        assert metrics["est_fps"] >= target_fps
    else:
        # no target: the optimizer maximizes the throughput within the budget
        metrics = run_feature_build(board, "fold-max", **OPTIMIZER_FOLDING)
        if metrics["est_vs_ref"] < 1:
            warnings.warn("maximized folding is %.2fx the committed one" % metrics["est_vs_ref"])


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.dwc
@pytest.mark.xdist_group("bert")
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_bert_dwc(board):
    target_fps = int(estimated_throughput(reference_estimate(board)))
    # the folding search is cheap: find a seed whose padded folding pads (padding wins
    # only on some search paths), skip the hardware build if none does
    for seed in DWC_SEEDS:
        padded = dict(
            OPTIMIZER_FOLDING,
            target_fps=target_fps,
            folding_maximum_padding=FOLDING_MAX_PADDING,
            folding_seed=seed,
        )
        if run_feature_build(board, "dwc-est", estimate_only=True, **padded)["pad_dwcs"]:
            break
    else:
        pytest.skip("no padding needed: no padded folding inserts a padding DWC")
    metrics = run_feature_build(board, "dwc", **padded)
    assert metrics["est_fps"] >= target_fps
