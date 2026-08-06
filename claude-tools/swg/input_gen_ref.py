# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Cycle-accurate Python reference for ``finn-rtllib/mvu_tiled/input_gen.sv``.

A transliteration of the RTL, driven with ``ivld`` always high and ``ordy``
always high -- the same stimulus the FIFO characterisation uses. It is the
ground truth a tree model for that module is written against, and it needs no
verilator, no vivado and no ipgen, so a candidate tree can be scored in
milliseconds instead of minutes.

The correspondence to the RTL is deliberately literal (same names, same
elaboration-time functions) so that a future change to the module can be
diffed against it. Where it departs: it counts transactions rather than moving
data, and it has no reset sequence beyond the initial state.

    from input_gen_ref import loop_nest_conv, simulate, tav
    dims, coefs, fm = loop_nest_conv(ifm_dim=(8, 8), k=(3, 3), stride=(1, 1),
                                     dilation=(1, 1), ifm_ch=4, simd=2)
    period, rd, wr = tav(dims, coefs, fm)
"""

import numpy as np


def init_w(dims, coefs, fm_size):
    return [fm_size] + list(coefs)


def init_r_flag(dims, coefs, w):
    d = len(dims)
    a = [True] + [False] * d
    for i in range(1, d + 1):
        a[i] = a[i - 1] and coefs[i - 1] > 0 and coefs[i - 1] * dims[i - 1] <= w[i - 1]
    return a


def init_rp_inc(dims, coefs, w):
    d = len(dims)
    a = [0] * (d + 1)
    rw = 0
    for i in range(d, -1, -1):
        if i < d:
            rw = (dims[i] - 1) * coefs[i] + rw
        a[i] = w[i] - rw
    return a


def init_fp_inc(dims, coefs, w, r_flag):
    d = len(dims)
    a = [0] * (d + 1)
    fw = 0
    for i in range(d, -1, -1):
        if i < d:
            fw = (dims[i] - 1) * coefs[i] + fw if r_flag[i + 1] else 0
        a[i] = (fw - w[i]) if r_flag[i] else 0
    return a


def max_occupancy(dims, coefs, rp_inc, r_flag):
    d = len(dims)
    m = 0
    for i in range(d):
        m = max(m, -rp_inc[i])
    rw = fw = 0
    for i in range(d - 1, -1, -1):
        rw = (dims[i] - 1) * coefs[i] + rw
        fw = (dims[i] - 1) * coefs[i] + fw if r_flag[i + 1] else 0
        m = max(m, rw - fw)
    return m


def buf_size(dims, coefs, fm_size):
    """``BUF_SIZE`` exactly as the module elaborates it: a power of two."""
    w = init_w(dims, coefs, fm_size)
    r_flag = init_r_flag(dims, coefs, w)
    rp_inc = init_rp_inc(dims, coefs, w)
    m = max_occupancy(dims, coefs, rp_inc, r_flag)
    addr_bits = max(1, int(np.ceil(np.log2(m + 1 + 2))))
    return 1 << addr_bits


def ptr_bits(dims, coefs, fm_size):
    """``PTR_BITS``: the width of ``Wp``, ``WpZ``, ``Rp`` and ``Cap``.

    The buffer, plus the largest terminal increment, plus a sign bit. The
    increment has to be in there because a nest that does not read all of its
    feature map releases the unread part in one lump when the frame completes,
    and that lump carries the free pointer past the write pointer: ``Cap`` --
    a counter, whose sign bit is ``irdy`` -- swings to ``-(BUF_SIZE-1) - lump``,
    and ``Rp - WpZ`` swings the same way. Sized for the buffer alone both wrap,
    which is the deadlock this workbench found and ``input_gen.sv`` has now been
    fixed for. Kept exactly in step with the RTL's own ``PTR_BITS``.
    """
    w = init_w(dims, coefs, fm_size)
    r_flag = init_r_flag(dims, coefs, w)
    rp_inc = init_rp_inc(dims, coefs, w)
    fp_inc = init_fp_inc(dims, coefs, w, r_flag)
    max_abs = max([abs(x) for x in rp_inc] + [abs(x) for x in fp_inc] + [0])
    return 1 + int(np.ceil(np.log2(buf_size(dims, coefs, fm_size) + max_abs + 1)))


def ptr(value, bits):
    """A value in ``ptr_t``, i.e. two's complement in ``bits`` bits."""
    mask = (1 << bits) - 1
    value &= mask
    return value - (mask + 1) if value > (mask >> 1) else value


def simulate(dims, coefs, fm_size, n_frames=4, max_cycles=None):
    """Run the module with input always valid and output always ready.

    Returns ``(reads, writes)``, two per-cycle 0/1 arrays: ``reads[c]`` is an
    input beat accepted on cycle ``c``, ``writes[c]`` an output beat produced.
    """
    d = len(dims)
    w = init_w(dims, coefs, fm_size)
    r_flag = init_r_flag(dims, coefs, w)
    rp_inc_t = init_rp_inc(dims, coefs, w)
    fp_inc_t = init_fp_inc(dims, coefs, w, r_flag)
    bs = buf_size(dims, coefs, fm_size)
    bits = ptr_bits(dims, coefs, fm_size)

    beats = int(np.prod(dims)) * n_frames
    limit = max_cycles or (4 * beats + 4 * fm_size * n_frames + 64)

    cnt = [dims[i] - 2 for i in range(d)]  # signed; -1 means done
    Wp = WpZ = Rp = 0
    Cap = ptr(-bs + 1, bits)
    OVld = False
    reads, writes = [], []
    produced = 0
    cycle = 0
    while produced < beats and cycle < limit:
        done = [(dims[i] == 1) or (cnt[i] < 0) for i in range(d)] + [True]
        term = [False] * (d + 1)
        term[d] = True
        for i in range(d - 1, -1, -1):
            term[i] = term[i + 1] and done[i]
        has_data = ptr(Rp - WpZ, bits) < 0
        ordy = True
        advance = has_data and ((not OVld) or ordy)
        irdy = Cap < 0

        rp_inc = fp_inc = 0
        for i in range(d, -1, -1):
            if term[i]:
                rp_inc = rp_inc_t[i]
                if r_flag[i]:
                    fp_inc = fp_inc_t[i]

        # an output beat leaves the module when the registered stage reloads
        out_beat = OVld and ordy
        reads.append(1 if irdy else 0)
        writes.append(1 if out_beat else 0)
        produced += out_beat

        istep = irdy  # ivld tied high
        WpZ_n = Wp
        Wp_n = ptr(Wp + istep, bits)
        Cap_n = ptr(Cap + (fp_inc if advance else 0) + istep, bits)
        Rp_n = ptr(Rp + (rp_inc if advance else 0), bits)
        OVld_n = has_data if ((not OVld) or ordy) else OVld

        if advance:
            for i in range(d - 1, -1, -1):
                step = term[i + 1]
                if step and dims[i] > 1:
                    cnt[i] = cnt[i] + ((dims[i] - 1) if cnt[i] < 0 else -1)

        Wp, WpZ, Rp, Cap, OVld = Wp_n, WpZ_n, Rp_n, Cap_n, OVld_n
        cycle += 1

    return np.array(reads, dtype=np.int8), np.array(writes, dtype=np.int8)


def settled_window(reads, writes, ends, fm_size, beats_per_frame):
    """``(start, period)`` of the shortest window that repeats and conserves tokens.

    Two things this has to get right.

    **A period is not always one frame.** A two-beat nest on a four-entry buffer
    settles at five cycles covering *two* frames -- the frames alternate 3 and 2
    cycles for ever -- and there is nothing wrong with that: it delivers its
    beats and reads its words, it just does not do it the same way twice running.
    So the span is searched from one frame upwards, shortest first.

    **Repetition alone is not enough.** While the writer is still spending the
    credit ``Cap`` started with, consecutive frames can be bit-identical -- every
    cycle a read -- and still not be the steady state, because what is draining
    is the credit rather than anything visible in the schedule. The invariant
    that settles it is conservation: a settled window reads exactly as many words
    as the feature maps it spans, and writes exactly as many beats.
    """
    for span in range(1, len(ends) // 2 + 1):
        # earliest qualifying window, not the latest: with a period spanning
        # several frames, which frame the window opens on is a rotation of the
        # whole schedule, and "the first one that settles" is the choice a model
        # walking forward from reset can also make
        for i in range(2 * span, len(ends)):
            period = ends[i] - ends[i - span]
            if period <= 0 or period != ends[i - span] - ends[i - 2 * span]:
                continue
            lo, mid = ends[i - 2 * span] + 1, ends[i - span] + 1
            if int(reads[mid : mid + period].sum()) != fm_size * span:
                continue
            if int(writes[mid : mid + period].sum()) != beats_per_frame * span:
                continue
            if np.array_equal(reads[lo:mid], reads[mid : mid + period]) and np.array_equal(
                writes[lo:mid], writes[mid : mid + period]
            ):
                return mid, period
    return None


def tav(dims, coefs, fm_size, n_frames=4, max_frames=256):
    """(period, cumulative reads, cumulative writes) of one steady-state frame.

    The period is measured between two frame completions, so start-up is
    excluded -- the same window rtlsim characterisation keeps. **How far in that
    window has to be is not fixed.** ``Cap`` starts at ``-BUF_SIZE+1``, which
    hands the writer ``BUF_SIZE-1`` words of credit it never gets back, and it
    spends that at one word per frame wherever the frame is one cycle longer
    than the words it consumes. A buffer bigger than a feature map -- every
    ``mvu_tiled`` nest, and any window whose feature map is small -- then takes
    tens of frames to drain it, and a window taken inside that transient reports
    one read per frame too many. That is a read-early bias, the direction that
    undersizes a FIFO, so the window is grown until two consecutive frames agree
    rather than trusted at a fixed depth.

    Returns ``None`` if the run never settles into a repeating frame.
    """
    beats_per_frame = int(np.prod(dims))
    while True:
        reads, writes = simulate(dims, coefs, fm_size, n_frames=n_frames)
        cw = np.cumsum(writes)
        if cw[-1] < beats_per_frame * n_frames:
            return None
        ends = [int(np.argmax(cw >= beats_per_frame * (i + 1))) for i in range(n_frames)]
        window = settled_window(reads, writes, ends, fm_size, beats_per_frame)
        if window is not None:
            lo, period = window
            break
        if n_frames >= max_frames:
            return None
        n_frames *= 2
    hi = lo + period
    r = np.cumsum(reads[lo:hi])
    w = np.cumsum(writes[lo:hi])
    return period, r, w


def loop_nest_conv(ifm_dim, k, stride, dilation, ifm_ch, simd, depthwise=0):
    """(DIMS, COEFS, FM_SIZE) for a sliding window, in ``input_gen`` terms.

    The output order is the one the MVAU expects: for each output pixel, walk
    the kernel, and inside the kernel walk the channel folds. Coefficients are
    in units of SIMD words, which is what the stream carries.

    ``FM_SIZE`` is the input stream period -- one feature map in SIMD words.
    Verify this mapping against the RTL instantiation in
    ``finn-rtllib/mvu_tiled/mvu_tiled_axi.sv`` before trusting it; it is the
    part of this file most likely to be wrong.
    """
    ifm_h, ifm_w = ifm_dim
    k_h, k_w = k
    s_h, s_w = stride
    d_h, d_w = dilation
    sf = ifm_ch // simd
    ofm_h = (ifm_h - ((k_h - 1) * d_h + 1)) // s_h + 1
    ofm_w = (ifm_w - ((k_w - 1) * d_w + 1)) // s_w + 1
    # dims outermost-first; coefs are the address step of each level, in words
    dims = [ofm_h, ofm_w, k_h, k_w, sf]
    coefs = [
        s_h * ifm_w * sf,
        s_w * sf,
        d_h * ifm_w * sf,
        d_w * sf,
        1,
    ]
    fm_size = ifm_h * ifm_w * sf
    if depthwise:
        # depthwise reorders the two innermost levels: channel fold outermost
        # of the window so each channel's window is contiguous
        dims = [ofm_h, ofm_w, sf, k_h, k_w]
        coefs = [s_h * ifm_w * sf, s_w * sf, 1, d_h * ifm_w * sf, d_w * sf]
    return dims, coefs, fm_size


if __name__ == "__main__":
    for name, kw in [
        ("3x3 s1 8x8 c4 simd2", dict(ifm_dim=(8, 8), k=(3, 3), stride=(1, 1), ifm_ch=4, simd=2)),
        (
            "3x3 s2 16x16 c8 simd8",
            dict(ifm_dim=(16, 16), k=(3, 3), stride=(2, 2), ifm_ch=8, simd=8),
        ),
        ("1x1 s1 8x8 c8 simd4", dict(ifm_dim=(8, 8), k=(1, 1), stride=(1, 1), ifm_ch=8, simd=4)),
    ]:
        dims, coefs, fm = loop_nest_conv(dilation=(1, 1), **kw)
        r = tav(dims, coefs, fm)
        print(
            "%-24s dims=%s coefs=%s fm=%d buf=%d -> %s"
            % (
                name,
                dims,
                coefs,
                fm,
                buf_size(dims, coefs, fm),
                (
                    "no settle"
                    if r is None
                    else "period %d, %d reads %d writes" % (r[0], r[1][-1], r[2][-1])
                ),
            )
        )
