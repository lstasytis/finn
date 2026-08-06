# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The SWG FSM, executed cycle by cycle -- the oracle the nest is scored against.

Lifted out of ``convolutioninputgenerator.py`` when the shipped model became an
analytical nest. It is exact by construction and far too slow and too flat to
ship, which is exactly what makes it a good reference.
"""

from qonnx.custom_op.general.im2col import compute_conv_output_dim  # noqa: F401

# Loop levels of swg_controller, innermost first. The counter cascade ripples
# in this order and STATE_LOOP_<name> names the level whose head increment is
# applied. The level above H is STATE_START, whose increment is zero.
SWG_LOOPS = ("SIMD", "KW", "KH", "W", "H")
SWG_START = -1


def swg_params(inst, impl_style):
    """The generated RTL parameters of ``inst``, as plain ints.

    Read back from ``prepare_codegen_default`` / ``prepare_codegen_parallel``
    rather than recomputed: those are what actually parameterise the Verilog,
    so taking them at the source is what keeps this model from drifting away
    from the hardware it claims to describe. Returns None if code generation
    declines the shape.
    """
    try:
        _, cg = getattr(inst, "prepare_codegen_" + impl_style)()
    except (AssertionError, AttributeError, KeyError, ValueError, ZeroDivisionError):
        return None
    p = {
        key.strip("$"): int(val[0])
        for key, val in cg.items()
        if len(val) == 1 and val[0].lstrip("-").isdigit()
    }
    p["INNERMOST_STATE"] = cg["$INNERMOST_STATE$"][0].replace("STATE_LOOP_", "")
    return p


class SwgController:
    """swg_controller: the five-deep counter nest that drives buffer addressing.

    ``advance()`` applies the head increment of the loop level that is ending
    and ripples the counters, as the always_ff does. ``addr_incr`` and
    ``tail_incr`` are combinational -- read them *before* advancing.
    """

    def __init__(self, p):
        self.limits = [p["LOOP_%s_ITERATIONS" % s] for s in SWG_LOOPS]
        self.head = [p["HEAD_INCR_%s" % s] for s in SWG_LOOPS]
        self.tails = (p["TAIL_INCR_W"], p["TAIL_INCR_H"], p["TAIL_INCR_LAST"])
        self.is_dw = p["IS_DEPTHWISE"]
        self.counters = list(self.limits)
        self.inner = self.state = SWG_LOOPS.index(p["INNERMOST_STATE"])

    @property
    def addr_incr(self):
        # STATE_START, the level above the outermost one, increments by nothing
        return self.head[self.state] if self.state != SWG_START else 0

    @property
    def tail_incr(self):
        c = self.counters  # levels 2, 3, 4 are KH, W, H
        if self.is_dw and c[2] >= 0:
            return 1
        return self.tails[0] if c[3] >= 0 else self.tails[1] if c[4] >= 0 else self.tails[2]

    def advance(self):
        c = self.counters
        if self.state != self.inner:
            self.state = self.inner
            return
        if c[0] < 0:
            # the innermost counter has run out, so the address increment passes
            # to the innermost level that still has iterations left; when none
            # has, the frame wraps
            self.state = next((lvl for lvl in range(1, len(SWG_LOOPS)) if c[lvl] >= 0), SWG_START)
        for lvl in range(len(SWG_LOOPS)):
            if c[lvl] >= 0:
                c[lvl] -= 1
                break
            c[lvl] = self.limits[lvl]


def swg_default_schedule(p, n_feature_maps=4, hard_limit=None):
    """Per-cycle (input transaction, output transaction) of the default-style SWG.

    Runs the FSM from reset with in0_V_V_TVALID and out_V_V_TREADY tied high.
    Returns ``(schedule, restarts)`` where ``restarts`` holds the cycle indices
    at which the generator wrapped round to the next feature map -- the period is
    the spacing between two of those, once the start-up transient is past.
    """
    LAST_READ, LAST_WRITE = p["LAST_READ_ELEM"], p["LAST_WRITE_ELEM"]
    BUF, EPW = p["BUF_ELEM_TOTAL"], p["ELEM_PER_WINDOW"]
    ctrl = SwgController(p)

    newest, current, first_next, pos_in_window = -1, 0, 0, 0
    fetching_done = write_cmd = writing_done = 0

    # a cycle bound that cannot be hit in a healthy configuration, so a bug in
    # the FSM transcription shows up as a bounded run rather than a hang
    if hard_limit is None:
        hard_limit = 64 * (LAST_READ + 1) * (EPW + 4) + 4096

    sched, restarts, cycle = [], [], 0
    while cycle < hard_limit and len(restarts) < n_feature_maps:
        write_ok = write_cmd  # out_V_V_TREADY tied high
        fetch_cmd = (current <= newest) and not fetching_done
        reading_done = newest == LAST_READ
        oldest = newest - (BUF - 1)
        read_ok = (not reading_done) and (
            fetching_done or (oldest < first_next and oldest < current)
        )
        sched.append((int(read_ok), int(write_ok)))

        # sequential block, in source order so that a later assignment to the
        # same register wins, as it does in the always_ff
        n_newest, n_current, n_first, n_pos = newest, current, first_next, pos_in_window
        n_fetching, n_write_cmd, n_writing = fetching_done, write_cmd, writing_done
        restarted = False

        if read_ok:
            n_newest = newest + 1
            if newest == LAST_READ - 1 and writing_done:
                n_newest, n_current, n_first, n_fetching, n_writing = -1, 0, 0, 0, 0
                restarted = True
        if fetch_cmd:
            n_pos = pos_in_window + 1 if pos_in_window != EPW - 1 else 0
            if pos_in_window == 0:
                n_first = first_next + ctrl.tail_incr
            if current == LAST_WRITE:
                n_fetching = 1
            else:
                n_current = current + ctrl.addr_incr
            n_write_cmd = 1
        if write_ok:
            n_write_cmd = 1 if fetch_cmd else 0
            if fetching_done:
                if reading_done or (read_ok and newest == LAST_READ - 1):
                    n_newest, n_current, n_first, n_fetching = -1, 0, 0, 0
                    restarted = True
                else:
                    n_writing = 1

        newest, current, first_next, pos_in_window = n_newest, n_current, n_first, n_pos
        fetching_done, write_cmd, writing_done = n_fetching, n_write_cmd, n_writing
        if fetch_cmd:
            ctrl.advance()
        if restarted:
            restarts.append(cycle)
        cycle += 1

    return sched, restarts


def swg_parallel_schedule(p, n_feature_maps=4, hard_limit=None):
    """Per-cycle (input transaction, output transaction) of the parallel-style SWG.

    Same convention as swg_default_schedule. With out_V_V_TREADY tied high the
    ``Write_done`` register can never set -- ``advance`` is asserted in every
    cycle in which ``write_ok`` is -- so the output transaction is simply
    ``write_cmd``.
    """
    LAST_READ, LAST_WRITE = p["LAST_READ_ELEM"], p["LAST_WRITE_ELEM"]
    FIRST_WRITE = p["FIRST_WRITE_ELEM"]
    ctrl = SwgController(p)

    newest, current, writing_done = -1, FIRST_WRITE, 0
    if hard_limit is None:
        hard_limit = 64 * (LAST_READ + 1) + 4096

    sched, restarts, cycle = [], [], 0
    while cycle < hard_limit and len(restarts) < n_feature_maps:
        write_ok = (current <= newest) and not writing_done
        reading_done = newest == LAST_READ
        read_ok = (not reading_done) and (writing_done or newest <= current)
        sched.append((int(read_ok), int(write_ok)))

        n_newest, n_current, n_writing = newest, current, writing_done
        restarted = False

        if read_ok:
            n_newest = newest + 1
            if newest == LAST_READ - 1 and writing_done:
                n_newest, n_current, n_writing = -1, FIRST_WRITE, 0
                restarted = True
        if write_ok:
            if current == LAST_WRITE:
                n_writing = 1
                if reading_done or (read_ok and newest == LAST_READ - 1):
                    n_newest, n_current, n_writing = -1, FIRST_WRITE, 0
                    restarted = True
            else:
                n_current = current + ctrl.addr_incr

        newest, current, writing_done = n_newest, n_current, n_writing
        if write_ok:
            # advance_controller is write_ok for the parallel style
            ctrl.advance()
        if restarted:
            restarts.append(cycle)
        cycle += 1

    return sched, restarts


