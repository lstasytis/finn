# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Folding flow: automatic folding at the throughput of the committed folding JSON.

Folds the cached model with SetFolding (the legacy ``naive`` folder and the
resource-aware ``optimizer``) at a target_fps matched to the committed folding
JSON, checks that the folded accelerator computes the golden output in cppsim
and stitched-IP rtlsim, and records the time of step_target_fps_parallelization
and whether the target was met.
FIFO sizing stays on dev's default strategy so only the folding differs.
"""

import pytest

import inspect
from feature_flow_helpers import (
    assert_flow_correct,
    cached_checkpoint,
    feature_cases,
    folding_config_file,
    record_target,
    reference_cycles,
    run_flow,
)

from finn.transformation.fpgadataflow.set_folding import SetFolding

FOLDING_STYLES = ["naive", "optimizer"]


def folding_target(model_dir, model, board, clk_ns):
    """(target_fps, cycle budget) matching the committed folding JSON."""
    cache_dir = cached_checkpoint(model_dir, model, board)
    ref_cycles = reference_cycles(cache_dir, folding_config_file(model_dir, model, board))
    clk_hz = 10**9 / clk_ns
    target_fps = int(clk_hz // ref_cycles)
    return target_fps, int(clk_hz / target_fps)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.finn_examples
@pytest.mark.parametrize("style", FOLDING_STYLES)
@pytest.mark.parametrize("model_dir,model,board", feature_cases())
def test_folding_flow(model_dir, model, board, style):
    if "style" not in inspect.signature(SetFolding.__init__).parameters:
        pytest.skip("SetFolding has no folding styles on this tree")
    target_fps, budget = folding_target(model_dir, model, board, 10.0)
    row = run_flow(
        "folding_" + style,
        model_dir,
        model,
        board,
        folding_config_file=None,
        target_fps=target_fps,
        folding_style=style,
    )
    # a target from a hand-tuned JSON may not fit the device (the optimizer then stays
    # within budget), so meeting it is recorded, not asserted
    record_target(row, budget)
    assert_flow_correct(row)
