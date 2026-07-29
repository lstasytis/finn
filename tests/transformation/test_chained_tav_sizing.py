# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the chained-TAV FIFO sizing primitives.

These cover the four quantities the pass is built from, on inputs small enough
to check by hand. They deliberately do not build a FINN graph: the graph-level
behaviour is measured on hardware (see ci/experiments/sizing-log.md), whereas
these are the places where an off-by-one or a sign error would be invisible in
an end-to-end number.

Run with ``uv run --with pytest python -m pytest`` -- a plain ``uv run pytest``
imports a different FINN tree.
"""

import numpy as np

from finn.transformation.fpgadataflow.derive_characteristic import (
    _arrival_of,
    _burst_above_rate,
    _longest_read_run,
    _peak_occupancy,
    _peak_occupancy_periodic,
    _stream_curves,
)


def test_arrival_of_indexes_by_token_count_not_position():
    # token n arrives at time 10*n
    sched = np.array([10, 20, 30, 40])
    # cumulative counts: nothing, then one token, then three
    counts = np.array([0, 1, 3, 4])
    assert _arrival_of(sched, counts).tolist() == [0, 10, 30, 40]


def test_arrival_of_clamps_counts_beyond_the_schedule():
    sched = np.array([10, 20])
    assert _arrival_of(sched, np.array([5])).tolist() == [20]


def test_peak_occupancy_counts_tokens_in_flight():
    # producer writes 4 tokens at t=0,1,2,3; consumer reads them at t=10..13
    writes = np.array([0, 1, 2, 3])
    reads = np.array([10, 11, 12, 13])
    assert _peak_occupancy(writes, reads) == 4
    # consumer keeping up exactly one cycle behind never lets more than one
    # token accumulate
    assert _peak_occupancy(np.array([0, 2, 4]), np.array([1, 3, 5])) == 1


def test_peak_occupancy_is_zero_for_a_producer_that_never_runs_ahead():
    writes = np.array([10, 20, 30])
    reads = np.array([10, 20, 30])
    # side="right" counts a same-cycle read as already taken
    assert _peak_occupancy(writes, reads) == 0


def test_peak_occupancy_handles_an_empty_edge():
    assert _peak_occupancy(np.array([]), np.array([])) == 0


def test_burst_above_rate_is_the_largest_excursion_over_the_rate_line():
    # 10 tokens read back to back at t=0..9, then idle; average rate 0.1/cycle.
    # By the last read, demand has reached 10 while the supply has delivered
    # 0.9 -- but the excursion is measured from the first read, which is served
    # out of the supply itself, so the buffer holds the other 9.
    reads = np.arange(10)
    assert _burst_above_rate(reads, 0.1) == 9
    # a supply that keeps up token for token needs no buffer at all
    assert _burst_above_rate(reads, 1.0) == 0


def test_burst_above_rate_degenerate_inputs():
    assert _burst_above_rate(np.array([]), 0.5) == 0
    assert _burst_above_rate(np.arange(4), 0.0) == 0


def test_longest_read_run_finds_the_back_to_back_stretch():
    # runs of 3, then 5, then 1
    reads = np.array([0, 1, 2, 10, 11, 12, 13, 14, 30])
    assert _longest_read_run(reads) == 5


def test_longest_read_run_degenerate_inputs():
    assert _longest_read_run(np.array([])) == 0
    assert _longest_read_run(np.array([7])) == 1
    assert _longest_read_run(np.array([0, 5, 10])) == 1


def test_burst_above_rate_never_exceeds_the_token_count():
    reads = np.arange(0, 100, 7)
    assert 0 < _burst_above_rate(reads, 0.01) <= len(reads)


# ---- steady-state (periodic) occupancy ---------------------------------------
#
# The pass sees one frame of each schedule at a time, but the design runs frames
# back to back forever. These cover the arithmetic that turns the first into the
# second, which is what lets a depth exceed a frame's tokens at all.


def _two_frames(times, per_frame, period):
    """A two-frame schedule: the same shape repeated one period later."""
    return np.concatenate([np.asarray(times), np.asarray(times) + period])


def test_periodic_occupancy_matches_the_one_shot_answer_when_nothing_carries_over():
    # producer bursts 4 tokens at t=0..3, consumer drains them by t=9, period 10:
    # nothing of frame f is still in flight when frame f+1 starts
    w = _two_frames([0, 1, 2, 3], 4, 10)
    r = _two_frames([6, 7, 8, 9], 4, 10)
    assert _peak_occupancy_periodic(w, r, 10, per_frame=4) == 4


def test_periodic_occupancy_carries_a_lagging_consumer_across_frames():
    # the consumer reads frame f only while frame f+1 is being written, so in
    # steady state nearly two frames are in flight at once -- the case a
    # single-frame measurement cannot express
    w = _two_frames([0, 1, 2, 3], 4, 10)
    r = _two_frames([12, 13, 14, 15], 4, 10)
    # the consumer lags by 1.2 frames, so at the worst moment frame f is fully
    # written and unread (4) while 2 tokens of frame f-1 are still queued: 6.
    # A single frame in isolation can never report more than its 4 tokens.
    assert _peak_occupancy_periodic(w, r, 10, per_frame=4) == 6
    assert _peak_occupancy(w[:4], r[:4]) == 4


def test_periodic_occupancy_ignores_a_second_frame_of_supplied_tokens():
    # both frames are present in the input, and counting them as one long frame
    # would double the answer
    w = _two_frames([0, 1, 2, 3], 4, 10)
    r = _two_frames([4, 5, 6, 7], 4, 10)
    assert _peak_occupancy_periodic(w, r, 10, per_frame=4) == 4


def test_periodic_occupancy_falls_back_without_a_period():
    w = np.array([0, 1, 2, 3])
    r = np.array([10, 11, 12, 13])
    assert _peak_occupancy_periodic(w, r, 0) == _peak_occupancy(w, r)


def test_periodic_occupancy_handles_an_empty_edge():
    assert _peak_occupancy_periodic(np.array([]), np.array([]), 10) == 0


# ---- binding token access vector rows to streams ------------------------------


class _FakeInst:
    """Just enough of a HWCustomOp for _stream_curves: per-input stream widths."""

    def __init__(self, widths):
        self._widths = widths

    def get_instream_width(self, i=0):
        return self._widths[i]


class _FakeNode:
    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs


def test_stream_curves_binds_one_row_per_streamed_input():
    # three streamed inputs and two constant ones, as ScaledDotProductAttention
    # has: Q/K/V are traced, the two threshold inputs are not
    node = _FakeNode(["q", "k", "v", "thr0", "thr1"], ["out"])
    inst = _FakeInst([64, 64, 64, 0, 0])
    rows_in = np.array([[0, 1], [0, 2], [0, 3]])
    rows_out = np.array([[0, 4]])
    in_curves, out_curves, bound = _stream_curves(node, inst, rows_in, rows_out)
    assert bound
    assert in_curves["q"].tolist() == [0, 1]
    assert in_curves["k"].tolist() == [0, 2]
    assert in_curves["v"].tolist() == [0, 3]
    assert "thr0" not in in_curves
    assert out_curves["out"].tolist() == [0, 4]


def test_stream_curves_falls_back_to_row_zero_when_the_counts_disagree():
    # a tree model can only ever emit one row, whatever the stream count, so a
    # multi-stream node derived that way must degrade to the old behaviour
    # rather than bind rows to the wrong tensors
    node = _FakeNode(["a", "b"], ["out0", "out1"])
    inst = _FakeInst([32, 32])
    rows_in = np.array([[0, 7]])
    rows_out = np.array([[0, 9]])
    in_curves, out_curves, bound = _stream_curves(node, inst, rows_in, rows_out)
    assert not bound
    assert in_curves["a"].tolist() == in_curves["b"].tolist() == [0, 7]
    assert out_curves["out0"].tolist() == out_curves["out1"].tolist() == [0, 9]


def test_stream_curves_binds_every_output_row():
    # StreamingSplit: one input, four outputs on their own schedules
    node = _FakeNode(["in"], ["o0", "o1", "o2", "o3"])
    inst = _FakeInst([4])
    rows_in = np.array([[0, 4]])
    rows_out = np.array([[0, 1], [0, 2], [0, 3], [0, 4]])
    _, out_curves, bound = _stream_curves(node, inst, rows_in, rows_out)
    assert bound
    assert [out_curves["o%d" % i][1] for i in range(4)] == [1, 2, 3, 4]


def test_experiment_knobs_default_to_the_shipped_behaviour():
    """Every ``CHAINED_TAV_*`` experiment knob must be a no-op out of the box.

    They exist so a sweep can subclass ``DeriveFIFOSizes`` and so the numbers in
    the sizing log are reproducible, not so anybody's build drifts. A knob whose
    default silently changed the pass would invalidate every board measurement
    recorded against it, and that is exactly the failure this asserts away.
    """
    from finn.transformation.fpgadataflow.derive_characteristic import DeriveFIFOSizes as D

    assert D.CHAINED_TAV_SLACK_SIDE == "chain"
    assert D.CHAINED_TAV_SLACK_SCALE == 1.0
    assert D.CHAINED_TAV_FLOOR_RATE == "graph"
    assert D.CHAINED_TAV_SMALL_PEAK == 0.0
    assert D.CHAINED_TAV_FRAMES == "both"
    assert D.CHAINED_TAV_CAP_GUARD == "down_chain"
    # and the shipped three, which older log entries quote by value
    assert D.CHAINED_TAV_SLACK_RELAXATION == 1.0
    assert D.CHAINED_TAV_FLOOR == "burst"
    assert D.CHAINED_TAV_THROTTLED_CAP == 256


def test_peak_occupancy_frames_selects_the_stored_frame():
    """``frames`` picks which of a TAV's two stored frames the peak comes from.

    Frame 0 is the one that fills the pipeline and frame 1 the first fully
    pipelined one; the shipped rule takes the larger. Build a two-frame edge
    whose run-ahead exists only in the second frame and check all three
    selections see what they should.
    """
    from finn.transformation.fpgadataflow.derive_characteristic import (
        _peak_occupancy_periodic,
    )

    period = 100
    # frame 0: writes and reads interleave, occupancy 1. frame 1: the producer
    # delivers all four up front, so occupancy reaches 4.
    w = np.array([0, 25, 50, 75, 100, 101, 102, 103])
    r = np.array([1, 26, 51, 76, 180, 181, 182, 183])
    fill = _peak_occupancy_periodic(w, r, period, per_frame=4, both_frames=False)
    steady = _peak_occupancy_periodic(w, r, period, per_frame=4, both_frames="steady")
    both = _peak_occupancy_periodic(w, r, period, per_frame=4, both_frames=True)
    assert steady > fill
    assert both == max(fill, steady)
