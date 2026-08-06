# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""A tree model for ``finn-rtllib/mvu_tiled/input_gen.sv``.

The module is a circular buffer between a linear input stream and an output
driven by a perfect loop nest. Driven the way FIFO characterisation drives it --
``ivld`` and ``ordy`` both tied high -- its whole behaviour is four lines of the
RTL, and they are two coupled max-plus recurrences::

    advance(k) = max(advance(k-1) + 1, accept(A[k]) + 2)
    accept(m)  = max(accept(m-1)  + 1, advance(gate[m]) + 1)

``advance(k)`` is the cycle the read pointer steps to output beat ``k`` (the
beat itself leaves one cycle later, because the output stage is registered);
``A[k]`` is the input word that beat reads, and the ``+2`` is ``Wp`` then
``WpZ``. ``accept(m)`` is the cycle input word ``m`` is taken; ``gate[m]`` is the
output beat whose level completion releases the buffer slot ``m`` will occupy,
and the ``+1`` is the registered ``Cap``. Everything else -- the sliding window
itself -- is in ``A`` and in the free-pointer staircase behind ``gate``, both of
which come straight out of the module's elaboration functions.

The two are solved **one loop iteration at a time**, not over a whole frame.
``block_pattern`` reads the nest and hands back the addresses and slot releases
of a single iteration of a level; ``Walk.step`` solves that one block by Kleene
iteration -- start at the schedule neither side stalls in, take prefix maxima,
repeat -- carrying forward only the accept times a later block can still read.
Consecutive blocks come out identical as soon as the pipeline has filled, and
``Walk.repeat`` then takes the rest of the level in one step. So a 330 000-cycle
frame costs a dozen block solves, its largest intermediate array is a few
thousand elements, and the period never exists as an array at all.

Scored against ``input_gen_ref.py``, the per-cycle transliteration of the same
RTL, the schedule is identical -- period, cumulative reads and cumulative
writes -- on every configuration of the sliding-window matrix.

    from input_gen_model import tree_model
    from input_gen_ref import loop_nest_conv
    dims, coefs, fm = loop_nest_conv((8, 8), (3, 3), (1, 1), (1, 1), 4, 2)
    node = tree_model(dims, coefs, fm)
"""

import collections
import math
import numpy as np

from finn.util.basic import Characteristic_Node

# The module's pipeline, in cycles. An accepted word is visible to the read side
# two cycles later (``Wp``, then ``WpZ``); a released slot is visible to the
# write side one cycle later (``Cap``); a beat leaves one cycle after the
# advance that fetched it (``OVld``/``OBuf``). This is the whole wind-up and the
# tree carries it itself -- there is no shift applied on top.
_ACCEPT_TO_VISIBLE = 2
_FREE_TO_VISIBLE = 1
_ADVANCE_TO_BEAT = 1

_MAX_PASSES = 8000
_MAX_FRAMES = 96


def nest_params(dims, coefs, fm_size):
    """``W``, ``R_FLAG``, ``TERMINAL_FP_INC`` and ``BUF_SIZE``, as elaborated.

    A transliteration of ``INIT_W``, ``INIT_R_FLAG``, ``INIT_RP_INC``,
    ``INIT_FP_INC`` and ``INIT_MAX_OCCUPANCY``. ``R_FLAG`` clears from the
    outside in: once a level fails ``COEFS[i-1]*DIMS[i-1] <= W[i-1]`` no inner
    level releases slots either, and ``TERMINAL_FP_INC`` is zero there.
    ``BUF_SIZE`` rounds up to a power of two, so the buffer is usually larger
    than the working set and that slack is what decouples the two sides.
    """
    d = len(dims)
    w = [fm_size] + list(coefs)
    r_flag = [True] + [False] * d
    for i in range(1, d + 1):
        r_flag[i] = r_flag[i - 1] and coefs[i - 1] > 0 and coefs[i - 1] * dims[i - 1] <= w[i - 1]
    rp_inc = [0] * (d + 1)
    fp_inc = [0] * (d + 1)
    rewind = free_rewind = 0
    for i in range(d, -1, -1):
        if i < d:
            rewind += (dims[i] - 1) * coefs[i]
            free_rewind = (dims[i] - 1) * coefs[i] + free_rewind if r_flag[i + 1] else 0
        rp_inc[i] = w[i] - rewind
        fp_inc[i] = (free_rewind - w[i]) if r_flag[i] else 0
    occupancy = max([0] + [-rp_inc[i] for i in range(d)])
    rewind = free_rewind = 0
    for i in range(d - 1, -1, -1):
        rewind += (dims[i] - 1) * coefs[i]
        free_rewind = (dims[i] - 1) * coefs[i] + free_rewind if r_flag[i + 1] else 0
        occupancy = max(occupancy, rewind - free_rewind)
    return w, r_flag, fp_inc, 1 << max(1, math.ceil(math.log2(occupancy + 3)))


def outer_level(dims):
    """The outermost level that is a loop. Levels above it run once."""
    for i, n in enumerate(dims):
        if n > 1:
            return i
    return len(dims) - 1


def block_pattern(dims, coefs, fp_inc, level):
    """One iteration of ``level``: its beats' addresses and slot releases.

    Straight off the nest, and the same for every iteration: block ``b`` of
    frame ``f`` reads ``addr + b*COEFS[level] + f*FM_SIZE``. Only the release on
    the block's last beat varies -- it completes ``level+1`` normally, and level
    0 on the frame's last block, since every level outside ``level`` runs once.
    """
    inner, icoef = list(dims[level + 1 :]), list(coefs[level + 1 :])
    beats = int(np.prod(inner)) if inner else 1
    beat = np.arange(beats, dtype=np.int64)
    addr = np.zeros(beats, dtype=np.int64)
    ends = np.full(beats, len(dims), dtype=np.int64)
    step = 1
    for j in range(len(inner) - 1, -1, -1):
        addr += icoef[j] * ((beat // step) % inner[j])
        step *= inner[j]
        ends[(beat + 1) % step == 0] = level + 1 + j
    freed = -np.array(fp_inc, dtype=np.int64)[ends]
    closing = freed.copy()
    closing[-1] = -fp_inc[0]
    return addr, freed, closing


class Walk:
    """One iteration of the outermost loop at a time, carrying only its state.

    The state between two blocks is small and bounded by the nest: the accept
    times of the words a later block can still read -- a window the width of the
    module's own buffer -- plus the last advance, the last accept, and how many
    slots have been released. Everything else is recomputed from ``dims`` and
    ``coefs``.
    """

    def __init__(self, dims, coefs, fm_size):
        _, _, fp_inc, self.buf = nest_params(dims, coefs, fm_size)
        self.level = outer_level(dims)
        self.addr, self.freed, self.closing = block_pattern(dims, coefs, fp_inc, self.level)
        self.blocks, self.coef, self.fm = dims[self.level], coefs[self.level], fm_size
        # the words the reset already leaves room for: taken back to back from
        # cycle 0, gated by nothing
        self.acc = np.arange(max(0, self.buf - 1), dtype=np.int64)
        self.base = 0  # word index ``acc[0]`` stands for
        self.last_adv = -1
        self.last_acc = self.acc.size - 1
        self.released = 0
        self.pending = self.acc.copy()  # accept cycles not yet inside a block
        self.index = 0
        self.end = -1  # cycle of the previous block's last output beat
        self.peak = self.acc.size

    def step(self):
        """Solve the next block, as a ``Block``, or ``None``.

        ``None`` is the decline. It covers a block that reads a word the free
        pointer has not released -- the module would deadlock -- and a block
        whose recurrences do not settle.
        """
        j, f = self.index % self.blocks, self.index // self.blocks
        freed = self.closing if j == self.blocks - 1 else self.freed
        addr = self.addr + j * self.coef + f * self.fm - self.base
        cumulative = np.cumsum(freed)
        first = max(self.released + self.buf - 1, self.base + self.acc.size)
        count = max(0, self.released + int(cumulative[-1]) + self.buf - 1 - first)
        gate = np.clip(
            np.searchsorted(
                cumulative, np.arange(count) + first - self.buf + 2 - self.released, "left"
            ),
            0,
            freed.size - 1,
        )
        times = np.concatenate((self.acc, np.zeros(count, dtype=np.int64)))
        if addr.min() < 0 or addr.max() >= times.size:
            return None  # reads a word no block has released: the module deadlocks
        self.peak = max(self.peak, times.size + freed.size + count)
        beat = np.arange(freed.size, dtype=np.int64)
        word = np.arange(count, dtype=np.int64)
        for _ in range(_MAX_PASSES):
            advance = beat + np.maximum.accumulate(
                np.maximum(times[addr] + _ACCEPT_TO_VISIBLE - beat, self.last_adv + 1 - beat)
            )
            if count == 0:
                break
            accept = word + np.maximum.accumulate(
                np.maximum(advance[gate] + _FREE_TO_VISIBLE - word, self.last_acc + 1 - word)
            )
            if np.array_equal(accept, times[times.size - count :]):
                break
            times[times.size - count :] = accept
        else:
            return None
        return self.close(times, advance, count, int(cumulative[-1]))

    def close(self, times, advance, count, frees):
        """Bank a solved block and roll the state on to the next one."""
        start = self.end + 1
        self.end = int(advance[-1]) + _ADVANCE_TO_BEAT
        queue = np.concatenate((self.pending, times[times.size - count :]))
        taken = queue <= self.end
        inside = queue[taken] - start
        self.pending = queue[~taken]
        self.acc = times
        self.last_adv = int(advance[-1])
        if count:
            self.last_acc = int(times[-1])
        self.released += frees
        self.index += 1
        drop = min(self.lowest_future_read() - self.base, self.acc.size - 1)
        if drop > 0:
            self.acc = self.acc[drop:]
            self.base += drop
        return Block(
            self.end - start + 1,
            tuple(inside.tolist()),
            tuple((advance + _ADVANCE_TO_BEAT - start).tolist()),
        )

    def lowest_future_read(self):
        """The lowest word index any block from here on will read.

        Accept times below it can be forgotten, and that is what bounds the
        carried state. Block ``b`` reads from ``(b % blocks)*COEFS[level] +
        (b // blocks)*FM_SIZE + min(addr)``, which is *not* monotone in ``b``:
        where ``(blocks-1)*COEFS[level]`` exceeds ``FM_SIZE`` -- a nest whose
        outer loop strides further than a feature map, so it reads into the next
        one -- the first block of the next frame reaches back behind the last
        block of this one. One frame of look-ahead settles it, because a whole
        frame later every read is exactly ``FM_SIZE`` higher.
        """
        span = range(self.index, self.index + self.blocks)
        return int(self.addr.min()) + min(
            (b % self.blocks) * self.coef + (b // self.blocks) * self.fm for b in span
        )

    def signature(self):
        """The state the next block will start from, relative to the last one.

        Everything the next ``step`` reads: where its addresses land in the
        accept window, how far the free pointer is ahead of that window, and the
        accept times themselves relative to the cycle the last block ended at.
        Two equal signatures mean two identical blocks, whatever their index.
        """
        j, f = self.index % self.blocks, self.index // self.blocks
        return (
            j * self.coef + f * self.fm - self.base,
            self.released - self.base,
            (self.acc - self.end).tobytes(),
            (self.pending - self.end).tobytes(),
            self.last_adv - self.end,
            self.last_acc - self.end,
        )

    def repeat(self, block, count, admitted, grew):
        """Take ``count`` more identical blocks in one step, by shifting the state."""
        duration = block.duration * count
        self.end += duration
        self.last_adv += duration
        self.last_acc += duration
        self.acc = self.acc + duration
        self.pending = self.pending + duration
        self.released += count * admitted
        self.base += count * grew
        self.index += count


def frame_blocks(dims, coefs, fm_size):
    """One settled period as ``[(count, block), ...]``, or ``None``.

    Walks blocks until the *state* at a frame boundary repeats one it has been
    in before; everything between the two is then the period. Matching on the
    state rather than on the blocks buys two things. It does not mistake the
    writer draining the credit the reset gave it for a steady state -- those
    frames are bit-identical while the accept queue behind them is still
    shortening. And it finds periods that span **several frames**: a two-beat
    nest on a four-entry buffer settles at five cycles covering two frames, the
    frames alternating three and two, which is a perfectly good steady state and
    not something to decline.

    Interior blocks stop changing long before any of that, so the walk
    fast-forwards over them by the same signature: what gets *solved* is a
    handful of blocks per frame rather than ``DIMS[level]`` of them, the period
    is only ever held run-length encoded, and no cycle array is ever assembled.
    """
    walk = Walk(dims, coefs, fm_size)
    frame, frames, seen, solved, previous = [], [], {}, 0, None
    while walk.index < _MAX_FRAMES * walk.blocks:
        was = (walk.released, walk.base)
        block = walk.step()
        if block is None:
            return None
        solved += 1
        admitted, grew = walk.released - was[0], walk.base - was[1]
        _append(frame, block, 1)
        here = walk.signature()
        j = walk.index % walk.blocks
        if 0 < j < walk.blocks - 1 and here == previous:
            ahead = walk.blocks - 1 - j
            walk.repeat(block, ahead, admitted, grew)
            frame[-1][0] += ahead
        previous = here
        if j == 0:
            frames.append(frame)
            frame = []
            if here in seen:
                period = []
                for one in frames[seen[here] + 1 :]:
                    for count, blk in one:
                        _append(period, blk, count)
                return [(n, b) for n, b in period], walk.peak, solved
            seen[here] = len(frames) - 1
    return None


def _append(runs, block, count):
    """Add ``count`` copies of a block to a run-length list."""
    if runs and runs[-1][1] == block:
        runs[-1][0] += count
    else:
        runs.append([count, block])


# One iteration of the derived loop: how many cycles it lasts, and which of them
# take an input word and which produce an output beat. Plain tuples, so two
# blocks compare equal when they are the same schedule.
Block = collections.namedtuple("Block", "duration reads writes")


def block_delta(block):
    """A block's cycles, as a ``(duration, 2)`` array of per-cycle read/write."""
    out = np.zeros((block.duration, 2), dtype=np.int64)
    out[list(block.reads), 0] = 1
    out[list(block.writes), 1] = 1
    return out


def _runs(delta, name):
    """A run-length leaf: the cycles of a block that has no loop left in it."""
    if delta.shape[0] == 0:
        return Characteristic_Node(name, [], True)
    cut = np.flatnonzero(np.any(np.diff(delta, axis=0) != 0, axis=1)) + 1
    start = np.concatenate(([0], cut))
    length = np.diff(np.concatenate((start, [delta.shape[0]])))
    return Characteristic_Node(
        name, [(int(a), [int(v[0]), int(v[1])]) for a, v in zip(length, delta[start])], True
    )


def _split(delta, n):
    """``n`` blocks, one per iteration of a level, or ``None`` if they do not cut.

    An iteration of level ``i`` emits the same number of beats as every other,
    so the cuts are at equal shares of the writes -- closed one cycle after the
    last write of the share, since that write is what ends it.
    """
    total = np.cumsum(delta[:, 1])
    if total[-1] == 0 or total[-1] % n:
        return None
    share = total[-1] // n
    end = np.searchsorted(total, share * np.arange(1, n + 1), side="left") + 1
    if end[-1] != delta.shape[0]:
        return None
    return [delta[(0 if j == 0 else end[j - 1]) : end[j]] for j in range(n)]


def _fold(delta, dims, level, name):
    """The nest's own loop structure inside one block, or a leaf where it stops.

    The outermost loop is derived, not folded -- ``frame_blocks`` builds it from
    ``dims`` without a period ever existing. **The levels inside a block are
    folded from that block's cycles**, which is the one place this model still
    cuts up a materialised trace. It is bounded: a block is one iteration of the
    outer loop, so the array is ``period / DIMS[level]`` cycles, not ``period``.
    Deriving these too would need the same carried state one level down, and the
    blocks there are small enough that the bookkeeping would cost more than the
    array does.

    Only the first iteration of a level differs, and only because the pipeline
    crosses the block boundary carrying the previous one's lead; every later one
    is bit-identical. So a level becomes ``[(1, head), (n-1, body)]``.
    """
    if level >= len(dims):
        return _runs(delta, name)
    if dims[level] < 2:
        return _fold(delta, dims, level + 1, name)  # a one-trip level is not a loop
    blocks = _split(delta, dims[level])
    if blocks is None or not all(np.array_equal(b, blocks[1]) for b in blocks[2:]):
        return _runs(delta, name)
    body = _fold(blocks[1], dims, level + 1, name)
    if np.array_equal(blocks[0], blocks[1]):
        return Characteristic_Node(name, [(dims[level], body)], False)
    head = _fold(blocks[0], dims, level + 1, name)
    return Characteristic_Node(name, [(1, head), (dims[level] - 1, body)], False)


def tree_model(dims, coefs, fm_size, name="input_gen nest"):
    """The steady-state schedule of one ``input_gen`` instance, or ``None``.

    ``None`` is reserved for nests that **cannot be built**, and it is meant to
    stay unreachable. Nothing in the sliding-window matrix, the ``mvu_tiled``
    instantiations, or 6000 random nests reaches it. What is left:

    * the free pointer not handing back exactly one frame of slots per frame.
      ``INIT_FP_INC`` telescopes to exactly that, so this is an invariant the
      elaboration guarantees rather than one it checks -- 40 000 random nests
      never reached it. It is kept so that a future change to ``INIT_FP_INC``
      that broke the invariant would decline rather than emit nonsense;
    * a block that reads a word the free pointer never releases, and recurrences
      or frames that do not settle inside ``_MAX_PASSES`` / ``_MAX_FRAMES``.

    Where it does fire the node falls back to rtlsim characterisation, which is
    ground truth: the conservative direction, since a wrong tree can undersize a
    FIFO and rtlsim cannot.
    """
    if len(dims) == 0 or any(x < 1 for x in dims) or fm_size < 1:
        return None
    _, _, fp_inc, _ = nest_params(dims, coefs, fm_size)
    level = outer_level(dims)
    _, freed, closing = block_pattern(dims, coefs, fp_inc, level)
    if (dims[level] - 1) * int(freed.sum()) + int(closing.sum()) != fm_size:
        return None  # the free pointer does not hand back a frame of slots
    settled = frame_blocks(dims, coefs, fm_size)
    if settled is None:
        return None
    inner = list(dims[outer_level(dims) + 1 :])
    return Characteristic_Node(
        name,
        [(count, _fold(block_delta(block), inner, 0, name)) for count, block in settled[0]],
        False,
    )
