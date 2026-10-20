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

import contextlib
import fcntl
import functools
import glob
import json
import numpy as np
import os
import shutil
import time
import warnings
from benchmark_helpers import (
    benchmark_root,
    check_build_outputs,
    find_cached_build,
    get_verify_steps,
    make_benchmark_cfg,
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


def configure_build(board, output_dir, small=False, **overrides):
    # Folding is hand-tuned for the VCK190; no specialize config, the default
    # layer specialization is used.
    folding = os.path.join(
        benchmark_root(), "bert", "folding_config", f"bert_folding_config_{board}.json"
    )
    cfg = dict(
        steps=bert_build_steps,
        generate_outputs=build_outputs,
        folding_config_file=small_folding(folding) if small else folding,
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


def reference_estimate(board, small=False):
    """Estimate-only build with the committed folding: its convert-phase
    checkpoint and estimated throughput are the feature tests' reference."""

    def build_estimate(output_dir):
        cfg = configure_build(
            board,
            output_dir,
            small=small,
            generate_outputs=ESTIMATE_OUTPUTS,
            stop_step=ESTIMATE_PHASE,
            verify_steps=[],
        )
        assert build.build_dataflow_cfg(model_file, cfg) == 0

    return cached_build(
        f"est_bert_{'small_' if small else ''}{board}_",
        build_estimate,
        "report/estimate_network_performance.json",
    )


def run_feature_build(small, board, name, estimate_only=False, **overrides):
    """Resume the reference checkpoint with ``overrides`` and, unless
    ``estimate_only``, check the stitched IP in rtlsim. Returns the metrics."""
    ref_dir = reference_estimate(board, small=small)
    output_dir = make_build_dir(f"{name}{'-small' if small else ''}_bert_{board}_")
    cfg = configure_build(
        board,
        output_dir,
        small=small,
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
@pytest.mark.parametrize("size", ["small", "full"])
@pytest.mark.parametrize("strategy", ["heuristic_analytical", "heuristic_rtlsim"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_bert_fifo_sizing(board, strategy, size):
    small = size == "small"
    run_feature_build(
        small, board, f"fifo-{strategy}", auto_fifo_depths=True, auto_fifo_strategy=strategy
    )


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.folding
@pytest.mark.xdist_group("bert")
@pytest.mark.parametrize("size", ["small", "full"])
@pytest.mark.parametrize("mode", ["target", "maximize", "fifo-aware"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_bert_folding(board, mode, size):
    small = size == "small"
    if small and mode == "maximize":
        pytest.skip("maximize does not use the committed folding")
    target_fps = int(estimated_throughput(reference_estimate(board, small=small)))
    if mode == "target":
        metrics = run_feature_build(
            small, board, "fold-target", target_fps=target_fps, **OPTIMIZER_FOLDING
        )
        assert metrics["est_fps"] >= target_fps
    elif mode == "fifo-aware":
        # optional, slower: the search scores foldings with FIFOs sized by the analytic sizer
        metrics = run_feature_build(
            small,
            board,
            "fold-fifoaware",
            target_fps=target_fps,
            enable_folding_fifo_heuristic=True,
            **OPTIMIZER_FOLDING,
        )
        assert metrics["est_fps"] >= target_fps
    else:
        # no target: the optimizer maximizes the throughput within the budget
        metrics = run_feature_build(small, board, "fold-max", **OPTIMIZER_FOLDING)
        if metrics["est_vs_ref"] < 1:
            warnings.warn("maximized folding is %.2fx the committed one" % metrics["est_vs_ref"])


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.dwc
@pytest.mark.xdist_group("bert")
@pytest.mark.parametrize("size", ["small", "full"])
@pytest.mark.parametrize("board", FEATURE_BUILDS)
def test_bert_dwc(board, size):
    small = size == "small"
    ref_dir = reference_estimate(board, small=small)
    target_fps = int(estimated_throughput(ref_dir))
    # the folding search is cheap: find a seed whose padded folding pads (padding wins
    # only on some search paths), skip the hardware build if none does
    for seed in DWC_SEEDS:
        padded = dict(
            OPTIMIZER_FOLDING,
            target_fps=target_fps,
            folding_maximum_padding=FOLDING_MAX_PADDING,
            folding_seed=seed,
        )
        if run_feature_build(small, board, "dwc-est", estimate_only=True, **padded)["pad_dwcs"]:
            break
    else:
        pytest.skip("no padding needed: no padded folding inserts a padding DWC")
    metrics = run_feature_build(small, board, "dwc", **padded)
    assert_target_or_trade(metrics, ref_dir, target_fps, board)
