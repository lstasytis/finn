# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""DWC padding flow: channel padding during folding, realized by padding/cropping DWCs.

Padding a node's channel count to a value with more divisors gives the folding
optimizer finer PE/SIMD choices; the DWCs between padded and unpadded neighbours
pad or crop each row. Two tests:

* test_padding_tightness: folds each model with the optimizer, with and without
  padding, for a sweep of throughput targets (multiples of the committed folding
  JSON's cycles) and for "maximize throughput within the device budget". DWCs are
  inserted before estimating, so their cost counts. Records how close each folding
  gets to its target and what it costs; padding must never make a folding worse.
* test_padding_flow: one padded folding per model, checked in folded cppsim and
  stitched-IP rtlsim against the golden output; skipped when the folding needed no
  padding.
"""

import pytest

import inspect
import json
import os
from feature_flow_helpers import (
    CACHE_PHASE,
    FEATURE_MODELS,
    assert_flow_correct,
    cached_checkpoint,
    feature_cases,
    folding_config_file,
    record_target,
    reference_cycles,
    run_flow,
)
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import part_map

MAX_PADDING = 4
CLK_NS = 10.0
# throughput targets as multiples of the committed folding JSON's cycles/frame
TARGET_FACTORS = [1, 1.5, 3, 7, 20]
# the optimizer's search effort as in a default build (folding_effort)
FOLDING_EFFORT = 50
SEED = 0
RESOURCES = ["LUT", "BRAM_18K", "URAM", "DSP"]
# padding widens the search space, so a padded folding may not be worse than this
COST_TOL = 0.02

# XNOR (1-bit) layers are never padded (a zero bit means -1 there), so the 2-bit
# BNN-PYNQ variants are added to exercise padded convolutions and FC layers
DWC_MODELS = FEATURE_MODELS + [("bnn-pynq", "tfc-w2a2"), ("bnn-pynq", "cnv-w2a2")]


def _require_padding():
    if not hasattr(StreamingDataWidthConverter, "pads_or_crops"):
        pytest.skip("DWC padding/cropping not available on this tree")
    if "folding_maximum_padding" not in inspect.signature(SetFolding.__init__).parameters:
        pytest.skip("channel padding needs the folding optimizer")


def fold(cache_dir, board, target_cycles, padding):
    """Fold the cached model, insert the DWCs and estimate cycles and resources."""
    model = ModelWrapper(os.path.join(cache_dir, "intermediate_models", CACHE_PHASE + ".onnx"))
    folder = SetFolding(
        target_cycles,
        platform=board,
        style="optimizer",
        folding_effort=FOLDING_EFFORT,
        folding_maximum_padding=padding,
        seed=SEED,
    )
    model = model.transform(folder)
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(part_map[board]))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateCycles())
    cycles = int(model.analysis(dataflow_performance)["max_cycles"])
    per_node = model.analysis(lambda m: res_estimation(m, part_map[board]))
    used = {r: sum(res.get(r, 0) for res in per_node.values()) for r in RESOURCES}
    budget = folder.last_search_report["budget"]
    utilization = max(used[r] / budget[r] for r in RESOURCES if budget.get(r, 0) > 1)
    dwcs = [n for n in model.graph.node if n.op_type.startswith("StreamingDataWidthConverter")]
    return {
        "cycles": cycles,
        "resources": used,
        "utilization": utilization,
        "fits_budget": folder.last_search_report["fits_budget"],
        "num_pad_crop_dwcs": sum(1 for n in dwcs if getCustomOp(n).pads_or_crops()),
    }


@pytest.mark.slow
@pytest.mark.finn_examples
@pytest.mark.parametrize("model_dir,model,board", feature_cases(DWC_MODELS))
def test_padding_tightness(model_dir, model, board):
    _require_padding()
    cache_dir = cached_checkpoint(model_dir, model, board)
    ref = reference_cycles(cache_dir, folding_config_file(model_dir, model, board))
    targets = [int(ref * f) for f in TARGET_FACTORS] + [None]

    rows = []
    for target in targets:
        plain = fold(cache_dir, board, target, 0)
        padded = fold(cache_dir, board, target, MAX_PADDING)
        rows.append({"target_cycles": target, "unpadded": plain, "padded": padded})

    results_dir = os.path.join(os.environ["FINN_BUILD_DIR"], "feature_flow_results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "padding_tightness.jsonl"), "a") as f:
        f.write(json.dumps({"model": model or model_dir, "board": board, "rows": rows}) + "\n")

    for row in rows:
        plain, padded, target = row["unpadded"], row["padded"], row["target_cycles"]
        if target is None:
            # maximize throughput within the budget: padding may not slow it down
            assert padded["cycles"] <= plain["cycles"] * (1 + COST_TOL), row
        elif plain["cycles"] <= target:
            # same target met: padding may not cost more
            assert padded["cycles"] <= target, row
            assert padded["utilization"] <= plain["utilization"] * (1 + COST_TOL), row


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("model_dir,model,board", feature_cases(DWC_MODELS))
def test_padding_flow(model_dir, model, board):
    _require_padding()
    cache_dir = cached_checkpoint(model_dir, model, board)
    ref = reference_cycles(cache_dir, folding_config_file(model_dir, model, board))
    target_fps = int((10**9 / CLK_NS) // ref)
    row = run_flow(
        "dwc_padding",
        model_dir,
        model,
        board,
        folding_config_file=None,
        target_fps=target_fps,
        folding_style="optimizer",
        folding_maximum_padding=MAX_PADDING,
    )
    record_target(row, int(10**9 / CLK_NS / target_fps))
    assert_flow_correct(row)
    if row["num_pad_crop_dwcs"] == 0:
        pytest.skip("the folding needed no channel padding, so no padding DWC was exercised")
