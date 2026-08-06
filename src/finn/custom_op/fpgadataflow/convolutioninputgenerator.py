# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import Characteristic_Node

# ONNX i/o tensor shape assumptions for ConvolutionInputGenerator:
# input 0 is the input tensor, shape NHWC = (1, IFMDim, IFMDim, IFMChannels)
# output 0 is the output tensor, shape NHWC:
#     = (1, OFMDim, OFMDim, (ConvKernelDim^2)*IFMChannels)


class ConvolutionInputGenerator(HWCustomOp):
    """Abstraction layer for HW implementation of ConvolutionInputGenerator"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "ConvKernelDim": ("ints", True, []),  # [H, W] = [Y, X]
            "IFMChannels": ("i", True, 0),
            "IFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "OFMDim": ("ints", True, []),  # [H, W] = [Y, X]
            "SIMD": ("i", True, 0),
            "Stride": ("ints", True, [1, 1]),  # [H, W] = [Y, X]
            # note: only dilation=1 supported for now
            "Dilation": ("ints", True, [1, 1]),  # [H, W] = [Y, X]
            # FINN DataTypes for inputs, weights, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            "depthwise": ("i", False, 0, {0, 1}),
            # FPGA resource type for ConvolutionInputGenerator input buffer
            # auto -- let Vivado HLS decide
            # block -- use BRAM
            # distributed -- use LUTRAM
            # ultra -- use URAM
            "ram_style": (
                "s",
                False,
                "distributed",
                {"auto", "block", "distributed", "ultra"},
            ),
            "parallel_window": ("i", False, 0, {0, 1}),
            # 1D (True) or 2D (False) spatial data
            "is1D": ("i", False, 0),
            # Enable reprogrammable implementation to change FM dimensions,
            # stride, or dilation during runtime (requires parallel_window = 0)
            "dynamic_mode": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        ishape = (1, ifm_dim_h, ifm_dim_w, ifm_ch)
        return ishape

    def get_folded_input_shape(self, ind=0):
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        simd = self.get_nodeattr("SIMD")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        wf = int(ifm_ch / simd)
        folded_ishape = (1, ifm_dim_h, ifm_dim_w, wf, simd)
        return folded_ishape

    def get_normal_output_shape(self, ind=0):
        k_h, k_w = self.get_nodeattr("ConvKernelDim")
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        stride_h, stride_w = self.get_nodeattr("Stride")
        dilation_h, dilation_w = self.get_nodeattr("Dilation")
        pad = 0
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, pad, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, pad, dilation_w)
        oshape = (1, ofm_dim_h, ofm_dim_w, k_h * k_w * ifm_ch)
        return oshape

    def get_folded_output_shape(self, ind=0):
        k_h, k_w = self.get_nodeattr("ConvKernelDim")
        ifm_dim_h, ifm_dim_w = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        stride_h, stride_w = self.get_nodeattr("Stride")
        dilation_h, dilation_w = self.get_nodeattr("Dilation")
        simd = self.get_nodeattr("SIMD")
        pad = 0
        ofm_dim_h = compute_conv_output_dim(ifm_dim_h, k_h, stride_h, pad, dilation_h)
        ofm_dim_w = compute_conv_output_dim(ifm_dim_w, k_w, stride_w, pad, dilation_w)
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        if self.use_parallel_window_output():
            wf = int((ifm_ch) // simd)
            folded_oshape = (1, ofm_dim_h, ofm_dim_w, wf, k_h * k_w * simd)
        else:
            wf = int((k_h * k_w * ifm_ch) // simd)
            folded_oshape = (1, ofm_dim_h, ofm_dim_w, wf, simd)
        return folded_oshape

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])

        # Test for changing input datatype
        if dtype != self.get_nodeattr("inputDataType"):
            # Issue a warning message
            warnings.warn(
                f"{node.name}: inputDataType changing from"
                f" {self.get_nodeattr('inputDataType')} to {dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("inputDataType", dtype.name)

        # Test for changing output datatype
        if dtype != self.get_nodeattr("outputDataType"):
            # Issue a warning message
            warnings.warn(
                f"{node.name}: outputDataType changing from"
                f" {self.get_nodeattr('outputDataType')} to {dtype}"
            )
            # Set the new datatype attribute
            self.set_nodeattr("outputDataType", dtype.name)
        # Propagate the datatype through the model graph
        model.set_tensor_datatype(node.output[0], dtype)

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def get_instream_width(self, ind=0):
        """Returns stream width, input and output stream width are equal for
        the sliding window function"""
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        ifm_ch = self.get_nodeattr("IFMChannels")
        assert ifm_ch % simd == 0, "SIMD must divide IFMChannels"
        in_width = simd * ibits
        return in_width

    def get_outstream_width(self, ind=0):
        if self.use_parallel_window_output():
            # feed all window pixels in parallel
            k_h, k_w = self.get_nodeattr("ConvKernelDim")
            return self.get_instream_width() * k_h * k_w
        else:
            # if parallel variant not in use: same width for output and input stream
            return self.get_instream_width()

    def get_1d_conv_attrs_normalized(self):
        # support both (1, D) and (D, 1) cases transparently:
        # For the kernel, presenting the input data of size D as
        # [H, W] = [Y, X] = [1, D] or [D, 1]
        # effectively gives the same result.
        # For consistency and ease of programming, this function
        # returns the attributes of the layer as follows:
        # [H, W] = [Y, X] = [1, D] or [D, 1] are always mapped to [1, D].
        # The dummy ('1') dimension is the Y-dimension.
        ifm_ch = self.get_nodeattr("IFMChannels")
        k = self.get_nodeattr("ConvKernelDim")
        ifm_dim = self.get_nodeattr("IFMDim")
        ofm_dim = self.get_nodeattr("OFMDim")
        stride = self.get_nodeattr("Stride")
        dilation = self.get_nodeattr("Dilation")

        # see defines() for an explanation
        if ifm_dim[1] == 1:
            ifm_dim = ifm_dim[::-1]
            ofm_dim = ofm_dim[::-1]
            k = k[::-1]
            stride = stride[::-1]
            dilation = dilation[::-1]

        return (ifm_ch, ifm_dim, ofm_dim, k, stride, dilation)

    def get_exp_cycles(self):
        return 0

    def bram_estimation(self):
        return 0

    def lut_estimation(self):
        return 0

    def uram_estimation(self):
        return 0

    def execute_node(self, context, graph):
        # using Im2Col node to calculate output
        node = self.onnx_node
        ifm_dim = self.get_nodeattr("IFMDim")
        k = self.get_nodeattr("ConvKernelDim")
        s = self.get_nodeattr("Stride")
        d = self.get_nodeattr("Dilation")
        ifm_ch = self.get_nodeattr("IFMChannels")
        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        im2col_node = helper.make_node(
            "Im2Col",
            [node.input[0]],
            [node.output[0]],
            domain="qonnx.custom_op.general",
            stride=[s[0], s[1]],
            kernel_size=[k[0], k[1]],
            dilations=[d[0], d[1]],
            input_shape="(1,{},{},{})".format(ifm_dim[0], ifm_dim[1], ifm_ch),
        )
        graph_im2col = helper.make_graph(
            nodes=[im2col_node],
            name="single-im2col-exec",
            inputs=[inp],
            outputs=[outp],
        )

        opset_imports = [helper.make_opsetid("qonnx.custom_op.general", 1)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_im2col = ModelWrapper(qonnx_make_model(graph_im2col, **onnx_kwargs))
        model_im2col.set_tensor_datatype(node.input[0], self.get_input_datatype())
        # use execution function from Im2Col node
        # this automatically updates the execution context
        inst = getCustomOp(im2col_node)
        inst.execute_node(context, model_im2col.graph)

    def get_tree_model(self):
        """The sliding-window generator as a loop nest, or None if not covered.

        ``swg_controller`` (finn-rtllib/swg/swg_common.sv) is a five-deep
        counter nest -- H, W, then the kernel and channel loops -- and the
        buffer around it is driven entirely by that nest. One output beat
        leaves per innermost iteration, and the free pointer releases a *draw*
        of input slots each time a level completes: ``TAIL_INCR_W`` at the end
        of a window, ``TAIL_INCR_H`` at the end of a row, ``TAIL_INCR_LAST`` at
        the end of the frame. Input words are taken back to back as each draw
        lands.

        So the schedule is the nest, and the tree states it as one: a frame of
        rows, a row of windows, a window of free-pointer steps. Nothing here
        walks a cycle. The three places the nest is not the whole story -- the
        buffer fill before the first beat, a row whose draw is too big to take
        inside the beats it has, and the drain of whatever is left when
        fetching finishes -- are the three extra blocks, one leaf each.

        Covers either RTL implementation style, ``dynamic_mode`` included: the
        dynamic template only makes the loop bounds writable over AXI-lite, and
        it powers up holding the same compile-time values this reads. Declines
        an HLS variant, whose schedule Vitis generates rather than this
        controller, and any shape code generation refuses.

        The ``claude-tools`` branch holds the FSM this replaced, kept as an
        oracle, and the harness that scores the nest against it.
        """
        if "_rtl" not in type(self).__name__:
            return None
        try:
            impl_style = self.select_impl_style()
        except (AttributeError, AssertionError):
            return None
        if impl_style not in ("default", "parallel"):
            return None
        ifm_ch, simd = self.get_nodeattr("IFMChannels"), self.get_nodeattr("SIMD")
        if simd <= 0 or ifm_ch % simd != 0:
            return None

        def leaf(name, runs):
            return Characteristic_Node(name, [(int(n), list(v)) for n, v in runs if n > 0], True)

        def comp(name, kids):
            return Characteristic_Node(name, [(int(n), c) for n, c in kids if n > 0 and c], False)

        def clip(v, lo, hi):
            return max(lo, min(v, hi))

        def params():
            # read back from the code generator rather than recomputed: these
            # are what parameterise the Verilog, so the model cannot drift
            try:
                _, cg = getattr(self, "prepare_codegen_" + impl_style)()
            except (AssertionError, AttributeError, KeyError, ValueError, ZeroDivisionError):
                return None
            p = {
                key.strip("$"): int(val[0])
                for key, val in cg.items()
                if len(val) == 1 and val[0].lstrip("-").isdigit()
            }
            p["INNERMOST_STATE"] = cg["$INNERMOST_STATE$"][0].replace("STATE_LOOP_", "")
            return p

        def dims(p):
            # LOOP_x_ITERATIONS is the counter's reload value, trips - 2; the
            # innermost level loses one more, the FSM starting in its state
            inner = p["INNERMOST_STATE"]
            return [
                p["LOOP_%s_ITERATIONS" % s] + 2 + (1 if s == inner else 0)
                for s in ("H", "W", "KH", "KW", "SIMD")
            ]

        def demand_lead(p):
            """How far the input must run ahead of the beats, in cycles.

            Beat ``j`` reads word ``addr[j]``, and ``addr`` is the nest at
            ``j``, so the frame runs ``max_j (addr[j] - j)`` longer than its
            beats alone would. That maximum takes every level as far as it
            goes, which makes it a sum rather than a search.

            Known short, and the only seam a better term has to fit: reads are
            throttled by the free pointer, so word ``addr[j]`` is not there at
            cycle ``addr[j]`` and the two waits compound instead of adding.
            """
            lead, inner = 0, 1
            for name, trips in zip(("SIMD", "KW", "KH", "W", "H"), reversed(dims(p))):
                lead += (trips - 1) * max(0, p["HEAD_INCR_" + name] - inner)
                inner *= trips
            return lead

        def default_nest(p):
            h, w, kh, kw, sd = dims(p)
            epw = p["ELEM_PER_WINDOW"]  # beats between two free-pointer steps
            beats = kh * kw * sd  # output beats per window
            if epw <= 0 or beats % epw or h < 1 or w < 1:
                return None
            steps = beats // epw  # steps per window: one, or the channel factor
            draw_w = (steps - 1) + p["TAIL_INCR_W"]
            draw_h = (steps - 1) + p["TAIL_INCR_H"]
            n_read = p["LAST_READ_ELEM"] + 1
            cap = w * beats

            def step(reads):
                return leaf("step", [(reads, [1, 1]), (epw - reads, [0, 1])])

            def window(name, reads):
                # one slot freed per channel step and the whole draw on the
                # last, so a bigger draw spreads back over the earlier steps
                reads = clip(reads, 0, beats)
                if steps == 1:
                    return step(reads)
                last = clip(reads - (steps - 1), 0, epw)
                q, r = divmod(reads - last, steps - 1)
                return comp(name, [(r, step(q + 1)), (steps - 1 - r, step(q)), (1, step(last))])

            def row(name, total, debt=0):
                total = max(0, total)
                inside = min(total, cap)
                a = clip(draw_w if per_row < cap else beats, 0, beats)
                if w > 1:
                    if (w - 1) * a + beats < inside:
                        a = clip(-(-(inside - beats) // (w - 1)), 0, beats)
                    elif (w - 1) * a > inside:
                        a = inside // (w - 1)
                end = clip(inside - (w - 1) * a, 0, beats)
                placed = (w - 1) * a + end
                spill = clip(total - placed, 0, stall)
                d = clip(debt, 0, beats - end)
                tail = window("win_h", end)
                if d:
                    # beats this window gives back to the lead-in, which fired
                    # them early while waiting for their words
                    tail = leaf(
                        "win_h_debt", [(end, [1, 1]), (beats - end - d, [0, 1]), (d, [0, 0])]
                    )
                node = comp(
                    name,
                    [
                        (w - 1, window("win", a)),
                        (1, tail),
                        (1, leaf("stall", [(spill, [1, 0]), (stall - spill, [0, 0])])),
                    ],
                )
                return node, placed + spill, d

            per_row = min((w - 1) * draw_w + draw_h, cap)
            # a row whose draw does not fit inside its beats stays one
            # free-pointer step behind for the whole row
            stall = beats - epw if (per_row + beats - epw > cap or draw_h > beats) else 0
            # the frame waits for whichever comes later: a whole window in the
            # buffer, or the read pointer far enough ahead that the first
            # window's last beat has its word
            windup = max(demand_lead(p), p["TAIL_INCR_LAST"] + 3 - beats)
            lead = max(0, p["BUF_ELEM_TOTAL"] - 1 - windup)
            budget = max(0, n_read - windup)
            # the lead-in is not idle: the beats whose words it waits for fire
            # inside it, borrowed from the last window so the writes balance
            paced = clip(epw - 1, 0, min(windup, beats - 1))
            # a single-row frame has no later row to carry what its draws leave
            first, n_first, _ = row(
                "row_first",
                budget if h == 1 else min(per_row + lead, budget),
                debt=paced if h == 1 else 0,
            )
            mid, n_mid, _ = row("row", per_row)
            left = max(0, budget - n_first - max(0, h - 2) * n_mid)
            last, n_last, debt = (
                row("row_last", left, debt=paced) if h > 1 else (None, 0, min(paced, n_first))
            )
            if h == 1:
                debt = paced
            gap = (windup // debt - 1) if debt else 0
            return comp(
                "swg_nest",
                [
                    (
                        1,
                        comp(
                            "lead_in",
                            [
                                (debt, leaf("wait", [(gap, [1, 0]), (1, [1, 1])])),
                                (
                                    1,
                                    leaf(
                                        "wait_end",
                                        [(max(0, windup - debt * (gap + 1)), [1, 0])],
                                    ),
                                ),
                            ],
                        ),
                    ),
                    (1, first),
                    (h - 2, mid),
                    (1 if h > 1 else 0, last),
                    # the frame reads its feature map exactly once: what the
                    # draws left drains at one word per cycle
                    (1, leaf("drain", [(max(0, left - n_last), [1, 0])])),
                ],
            )

        def parallel_nest(p):
            # one beat carries the whole kernel, so the input stream paces the
            # frame and the nest shows in the gaps between beats: the head
            # increment of the level that ended is how far the pointer jumps
            h, w, kh, _, _ = dims(p)
            if h < 1 or w < 2:
                return None
            n_read = p["LAST_READ_ELEM"] + 1
            gap_w, gap_h = max(0, p["HEAD_INCR_W"] - 1), max(0, p["HEAD_INCR_H"] - 1)
            row_len = (w - 1) * (kh + gap_w) + kh + gap_h
            if row_len <= 0:
                return None
            fill = clip(n_read - h * row_len, 0, p["FIRST_WRITE_ELEM"] + 1)

            def rows(reading):
                beat = [1, 1] if reading else [0, 1]
                gap = [1, 0] if reading else [0, 0]
                return comp(
                    "row",
                    [
                        (w - 1, leaf("win", [(kh, beat), (gap_w, gap)])),
                        (1, leaf("win_h", [(kh, beat), (gap_h, gap)])),
                    ],
                )

            dense = min(h, (n_read - fill) // row_len)
            return comp(
                "swg_nest",
                [
                    (1, leaf("carry", [(1, [0, 0])])),
                    (1, leaf("fill", [(fill, [1, 0])])),
                    (dense, rows(True)),
                    (h - dense, rows(False)),
                    (1, leaf("pad", [(max(0, n_read - fill - h * row_len), [1, 0])])),
                ],
            )

        p = params()
        if p is None:
            return None
        return (default_nest if impl_style == "default" else parallel_nest)(p)


# ---------------------------------------------------------------------------
# A tree model for finn-rtllib/mvu_tiled/input_gen.sv, parked.
#
# input_gen is a generic loop-nest input generator and the intended replacement
# for this operator. The model below is complete and was validated cycle-exact
# against a Python transliteration of the RTL on the whole sliding-window
# configuration matrix, on the 192 nests mvu_tiled_axi actually instantiates,
# and on 6000 random nests -- 0 wrong, 0 declined. It is commented out because
# no FINN custom op instantiates input_gen as a standalone sliding-window
# operator yet: there is no node for it to hang off. When one exists, this
# becomes its get_tree_model with the conv -> (DIMS, COEFS, FM_SIZE) mapping in
# front of it.
#
# What it does not cover: parallel_window. That style's output word is k*k
# times its input word and the module has one DATA_WIDTH for both ports, so the
# hardware cannot express it -- 41% of the matrix. That, and a 14-33%
# throughput loss on strided depthwise windows, is why the CIG is still here.
#
# Driven the way FIFO characterisation drives it (ivld and ordy both tied
# high), the module is two coupled max-plus recurrences:
#
#     advance(k) = max(advance(k-1) + 1, accept(A[k]) + 2)
#     accept(m)  = max(accept(m-1)  + 1, advance(gate[m]) + 1)
#
# advance(k) is the cycle the read pointer steps to output beat k, the beat
# itself leaving one cycle later because the output stage is registered; A[k]
# is the word that beat reads and the +2 is Wp then WpZ. accept(m) is the cycle
# input word m is taken; gate[m] is the beat whose level completion releases
# the slot m will occupy, and the +1 is the registered Cap. The sliding window
# itself lives in A and in the free-pointer staircase behind gate, both of
# which come out of the module's own elaboration functions.
#
# The pair is solved one loop iteration at a time rather than over a frame, so
# a 330 000-cycle frame costs about a dozen block solves and the period never
# exists as an array.
#
# The reference it was scored against, the RTL check that found the ptr_t
# deadlock, and the harnesses are on the claude-tools branch.
# ---------------------------------------------------------------------------
#
# import collections
# import math
# import numpy as np
#
# from finn.util.basic import Characteristic_Node
#
# # The module's pipeline, in cycles. An accepted word is visible to the read side
# # two cycles later (``Wp``, then ``WpZ``); a released slot is visible to the
# # write side one cycle later (``Cap``); a beat leaves one cycle after the
# # advance that fetched it (``OVld``/``OBuf``). This is the whole wind-up and the
# # tree carries it itself -- there is no shift applied on top.
# _ACCEPT_TO_VISIBLE = 2
# _FREE_TO_VISIBLE = 1
# _ADVANCE_TO_BEAT = 1
#
# _MAX_PASSES = 8000
# _MAX_FRAMES = 96
#
#
# def nest_params(dims, coefs, fm_size):
#     """``W``, ``R_FLAG``, ``TERMINAL_FP_INC`` and ``BUF_SIZE``, as elaborated.
#
#     A transliteration of ``INIT_W``, ``INIT_R_FLAG``, ``INIT_RP_INC``,
#     ``INIT_FP_INC`` and ``INIT_MAX_OCCUPANCY``. ``R_FLAG`` clears from the
#     outside in: once a level fails ``COEFS[i-1]*DIMS[i-1] <= W[i-1]`` no inner
#     level releases slots either, and ``TERMINAL_FP_INC`` is zero there.
#     ``BUF_SIZE`` rounds up to a power of two, so the buffer is usually larger
#     than the working set and that slack is what decouples the two sides.
#     """
#     d = len(dims)
#     w = [fm_size] + list(coefs)
#     r_flag = [True] + [False] * d
#     for i in range(1, d + 1):
#         r_flag[i] = r_flag[i - 1] and coefs[i - 1] > 0 and coefs[i - 1] * dims[i - 1] <= w[i - 1]
#     rp_inc = [0] * (d + 1)
#     fp_inc = [0] * (d + 1)
#     rewind = free_rewind = 0
#     for i in range(d, -1, -1):
#         if i < d:
#             rewind += (dims[i] - 1) * coefs[i]
#             free_rewind = (dims[i] - 1) * coefs[i] + free_rewind if r_flag[i + 1] else 0
#         rp_inc[i] = w[i] - rewind
#         fp_inc[i] = (free_rewind - w[i]) if r_flag[i] else 0
#     occupancy = max([0] + [-rp_inc[i] for i in range(d)])
#     rewind = free_rewind = 0
#     for i in range(d - 1, -1, -1):
#         rewind += (dims[i] - 1) * coefs[i]
#         free_rewind = (dims[i] - 1) * coefs[i] + free_rewind if r_flag[i + 1] else 0
#         occupancy = max(occupancy, rewind - free_rewind)
#     return w, r_flag, fp_inc, 1 << max(1, math.ceil(math.log2(occupancy + 3)))
#
#
# def outer_level(dims):
#     """The outermost level that is a loop. Levels above it run once."""
#     for i, n in enumerate(dims):
#         if n > 1:
#             return i
#     return len(dims) - 1
#
#
# def block_pattern(dims, coefs, fp_inc, level):
#     """One iteration of ``level``: its beats' addresses and slot releases.
#
#     Straight off the nest, and the same for every iteration: block ``b`` of
#     frame ``f`` reads ``addr + b*COEFS[level] + f*FM_SIZE``. Only the release on
#     the block's last beat varies -- it completes ``level+1`` normally, and level
#     0 on the frame's last block, since every level outside ``level`` runs once.
#     """
#     inner, icoef = list(dims[level + 1 :]), list(coefs[level + 1 :])
#     beats = int(np.prod(inner)) if inner else 1
#     beat = np.arange(beats, dtype=np.int64)
#     addr = np.zeros(beats, dtype=np.int64)
#     ends = np.full(beats, len(dims), dtype=np.int64)
#     step = 1
#     for j in range(len(inner) - 1, -1, -1):
#         addr += icoef[j] * ((beat // step) % inner[j])
#         step *= inner[j]
#         ends[(beat + 1) % step == 0] = level + 1 + j
#     freed = -np.array(fp_inc, dtype=np.int64)[ends]
#     closing = freed.copy()
#     closing[-1] = -fp_inc[0]
#     return addr, freed, closing
#
#
# class Walk:
#     """One iteration of the outermost loop at a time, carrying only its state.
#
#     The state between two blocks is small and bounded by the nest: the accept
#     times of the words a later block can still read -- a window the width of the
#     module's own buffer -- plus the last advance, the last accept, and how many
#     slots have been released. Everything else is recomputed from ``dims`` and
#     ``coefs``.
#     """
#
#     def __init__(self, dims, coefs, fm_size):
#         _, _, fp_inc, self.buf = nest_params(dims, coefs, fm_size)
#         self.level = outer_level(dims)
#         self.addr, self.freed, self.closing = block_pattern(dims, coefs, fp_inc, self.level)
#         self.blocks, self.coef, self.fm = dims[self.level], coefs[self.level], fm_size
#         # the words the reset already leaves room for: taken back to back from
#         # cycle 0, gated by nothing
#         self.acc = np.arange(max(0, self.buf - 1), dtype=np.int64)
#         self.base = 0  # word index ``acc[0]`` stands for
#         self.last_adv = -1
#         self.last_acc = self.acc.size - 1
#         self.released = 0
#         self.pending = self.acc.copy()  # accept cycles not yet inside a block
#         self.index = 0
#         self.end = -1  # cycle of the previous block's last output beat
#         self.peak = self.acc.size
#
#     def step(self):
#         """Solve the next block, as a ``Block``, or ``None``.
#
#         ``None`` is the decline. It covers a block that reads a word the free
#         pointer has not released -- the module would deadlock -- and a block
#         whose recurrences do not settle.
#         """
#         j, f = self.index % self.blocks, self.index // self.blocks
#         freed = self.closing if j == self.blocks - 1 else self.freed
#         addr = self.addr + j * self.coef + f * self.fm - self.base
#         cumulative = np.cumsum(freed)
#         first = max(self.released + self.buf - 1, self.base + self.acc.size)
#         count = max(0, self.released + int(cumulative[-1]) + self.buf - 1 - first)
#         gate = np.clip(
#             np.searchsorted(
#                 cumulative, np.arange(count) + first - self.buf + 2 - self.released, "left"
#             ),
#             0,
#             freed.size - 1,
#         )
#         times = np.concatenate((self.acc, np.zeros(count, dtype=np.int64)))
#         if addr.min() < 0 or addr.max() >= times.size:
#             return None  # reads a word no block has released: the module deadlocks
#         self.peak = max(self.peak, times.size + freed.size + count)
#         beat = np.arange(freed.size, dtype=np.int64)
#         word = np.arange(count, dtype=np.int64)
#         for _ in range(_MAX_PASSES):
#             advance = beat + np.maximum.accumulate(
#                 np.maximum(times[addr] + _ACCEPT_TO_VISIBLE - beat, self.last_adv + 1 - beat)
#             )
#             if count == 0:
#                 break
#             accept = word + np.maximum.accumulate(
#                 np.maximum(advance[gate] + _FREE_TO_VISIBLE - word, self.last_acc + 1 - word)
#             )
#             if np.array_equal(accept, times[times.size - count :]):
#                 break
#             times[times.size - count :] = accept
#         else:
#             return None
#         return self.close(times, advance, count, int(cumulative[-1]))
#
#     def close(self, times, advance, count, frees):
#         """Bank a solved block and roll the state on to the next one."""
#         start = self.end + 1
#         self.end = int(advance[-1]) + _ADVANCE_TO_BEAT
#         queue = np.concatenate((self.pending, times[times.size - count :]))
#         taken = queue <= self.end
#         inside = queue[taken] - start
#         self.pending = queue[~taken]
#         self.acc = times
#         self.last_adv = int(advance[-1])
#         if count:
#             self.last_acc = int(times[-1])
#         self.released += frees
#         self.index += 1
#         drop = min(self.lowest_future_read() - self.base, self.acc.size - 1)
#         if drop > 0:
#             self.acc = self.acc[drop:]
#             self.base += drop
#         return Block(
#             self.end - start + 1,
#             tuple(inside.tolist()),
#             tuple((advance + _ADVANCE_TO_BEAT - start).tolist()),
#         )
#
#     def lowest_future_read(self):
#         """The lowest word index any block from here on will read.
#
#         Accept times below it can be forgotten, and that is what bounds the
#         carried state. Block ``b`` reads from ``(b % blocks)*COEFS[level] +
#         (b // blocks)*FM_SIZE + min(addr)``, which is *not* monotone in ``b``:
#         where ``(blocks-1)*COEFS[level]`` exceeds ``FM_SIZE`` -- a nest whose
#         outer loop strides further than a feature map, so it reads into the next
#         one -- the first block of the next frame reaches back behind the last
#         block of this one. One frame of look-ahead settles it, because a whole
#         frame later every read is exactly ``FM_SIZE`` higher.
#         """
#         span = range(self.index, self.index + self.blocks)
#         return int(self.addr.min()) + min(
#             (b % self.blocks) * self.coef + (b // self.blocks) * self.fm for b in span
#         )
#
#     def signature(self):
#         """The state the next block will start from, relative to the last one.
#
#         Everything the next ``step`` reads: where its addresses land in the
#         accept window, how far the free pointer is ahead of that window, and the
#         accept times themselves relative to the cycle the last block ended at.
#         Two equal signatures mean two identical blocks, whatever their index.
#         """
#         j, f = self.index % self.blocks, self.index // self.blocks
#         return (
#             j * self.coef + f * self.fm - self.base,
#             self.released - self.base,
#             (self.acc - self.end).tobytes(),
#             (self.pending - self.end).tobytes(),
#             self.last_adv - self.end,
#             self.last_acc - self.end,
#         )
#
#     def repeat(self, block, count, admitted, grew):
#         """Take ``count`` more identical blocks in one step, by shifting the state."""
#         duration = block.duration * count
#         self.end += duration
#         self.last_adv += duration
#         self.last_acc += duration
#         self.acc = self.acc + duration
#         self.pending = self.pending + duration
#         self.released += count * admitted
#         self.base += count * grew
#         self.index += count
#
#
# def frame_blocks(dims, coefs, fm_size):
#     """One settled period as ``[(count, block), ...]``, or ``None``.
#
#     Walks blocks until the *state* at a frame boundary repeats one it has been
#     in before; everything between the two is then the period. Matching on the
#     state rather than on the blocks buys two things. It does not mistake the
#     writer draining the credit the reset gave it for a steady state -- those
#     frames are bit-identical while the accept queue behind them is still
#     shortening. And it finds periods that span **several frames**: a two-beat
#     nest on a four-entry buffer settles at five cycles covering two frames, the
#     frames alternating three and two, which is a perfectly good steady state and
#     not something to decline.
#
#     Interior blocks stop changing long before any of that, so the walk
#     fast-forwards over them by the same signature: what gets *solved* is a
#     handful of blocks per frame rather than ``DIMS[level]`` of them, the period
#     is only ever held run-length encoded, and no cycle array is ever assembled.
#     """
#     walk = Walk(dims, coefs, fm_size)
#     frame, frames, seen, solved, previous = [], [], {}, 0, None
#     while walk.index < _MAX_FRAMES * walk.blocks:
#         was = (walk.released, walk.base)
#         block = walk.step()
#         if block is None:
#             return None
#         solved += 1
#         admitted, grew = walk.released - was[0], walk.base - was[1]
#         _append(frame, block, 1)
#         here = walk.signature()
#         j = walk.index % walk.blocks
#         if 0 < j < walk.blocks - 1 and here == previous:
#             ahead = walk.blocks - 1 - j
#             walk.repeat(block, ahead, admitted, grew)
#             frame[-1][0] += ahead
#         previous = here
#         if j == 0:
#             frames.append(frame)
#             frame = []
#             if here in seen:
#                 period = []
#                 for one in frames[seen[here] + 1 :]:
#                     for count, blk in one:
#                         _append(period, blk, count)
#                 return [(n, b) for n, b in period], walk.peak, solved
#             seen[here] = len(frames) - 1
#     return None
#
#
# def _append(runs, block, count):
#     """Add ``count`` copies of a block to a run-length list."""
#     if runs and runs[-1][1] == block:
#         runs[-1][0] += count
#     else:
#         runs.append([count, block])
#
#
# # One iteration of the derived loop: how many cycles it lasts, and which of them
# # take an input word and which produce an output beat. Plain tuples, so two
# # blocks compare equal when they are the same schedule.
# Block = collections.namedtuple("Block", "duration reads writes")
#
#
# def block_delta(block):
#     """A block's cycles, as a ``(duration, 2)`` array of per-cycle read/write."""
#     out = np.zeros((block.duration, 2), dtype=np.int64)
#     out[list(block.reads), 0] = 1
#     out[list(block.writes), 1] = 1
#     return out
#
#
# def _runs(delta, name):
#     """A run-length leaf: the cycles of a block that has no loop left in it."""
#     if delta.shape[0] == 0:
#         return Characteristic_Node(name, [], True)
#     cut = np.flatnonzero(np.any(np.diff(delta, axis=0) != 0, axis=1)) + 1
#     start = np.concatenate(([0], cut))
#     length = np.diff(np.concatenate((start, [delta.shape[0]])))
#     return Characteristic_Node(
#         name, [(int(a), [int(v[0]), int(v[1])]) for a, v in zip(length, delta[start])], True
#     )
#
#
# def _split(delta, n):
#     """``n`` blocks, one per iteration of a level, or ``None`` if they do not cut.
#
#     An iteration of level ``i`` emits the same number of beats as every other,
#     so the cuts are at equal shares of the writes -- closed one cycle after the
#     last write of the share, since that write is what ends it.
#     """
#     total = np.cumsum(delta[:, 1])
#     if total[-1] == 0 or total[-1] % n:
#         return None
#     share = total[-1] // n
#     end = np.searchsorted(total, share * np.arange(1, n + 1), side="left") + 1
#     if end[-1] != delta.shape[0]:
#         return None
#     return [delta[(0 if j == 0 else end[j - 1]) : end[j]] for j in range(n)]
#
#
# def _fold(delta, dims, level, name):
#     """The nest's own loop structure inside one block, or a leaf where it stops.
#
#     The outermost loop is derived, not folded -- ``frame_blocks`` builds it from
#     ``dims`` without a period ever existing. **The levels inside a block are
#     folded from that block's cycles**, which is the one place this model still
#     cuts up a materialised trace. It is bounded: a block is one iteration of the
#     outer loop, so the array is ``period / DIMS[level]`` cycles, not ``period``.
#     Deriving these too would need the same carried state one level down, and the
#     blocks there are small enough that the bookkeeping would cost more than the
#     array does.
#
#     Only the first iteration of a level differs, and only because the pipeline
#     crosses the block boundary carrying the previous one's lead; every later one
#     is bit-identical. So a level becomes ``[(1, head), (n-1, body)]``.
#     """
#     if level >= len(dims):
#         return _runs(delta, name)
#     if dims[level] < 2:
#         return _fold(delta, dims, level + 1, name)  # a one-trip level is not a loop
#     blocks = _split(delta, dims[level])
#     if blocks is None or not all(np.array_equal(b, blocks[1]) for b in blocks[2:]):
#         return _runs(delta, name)
#     body = _fold(blocks[1], dims, level + 1, name)
#     if np.array_equal(blocks[0], blocks[1]):
#         return Characteristic_Node(name, [(dims[level], body)], False)
#     head = _fold(blocks[0], dims, level + 1, name)
#     return Characteristic_Node(name, [(1, head), (dims[level] - 1, body)], False)
#
#
# def tree_model(dims, coefs, fm_size, name="input_gen nest"):
#     """The steady-state schedule of one ``input_gen`` instance, or ``None``.
#
#     ``None`` is reserved for nests that **cannot be built**, and it is meant to
#     stay unreachable. Nothing in the sliding-window matrix, the ``mvu_tiled``
#     instantiations, or 6000 random nests reaches it. What is left:
#
#     * the free pointer not handing back exactly one frame of slots per frame.
#       ``INIT_FP_INC`` telescopes to exactly that, so this is an invariant the
#       elaboration guarantees rather than one it checks -- 40 000 random nests
#       never reached it. It is kept so that a future change to ``INIT_FP_INC``
#       that broke the invariant would decline rather than emit nonsense;
#     * a block that reads a word the free pointer never releases, and recurrences
#       or frames that do not settle inside ``_MAX_PASSES`` / ``_MAX_FRAMES``.
#
#     Where it does fire the node falls back to rtlsim characterisation, which is
#     ground truth: the conservative direction, since a wrong tree can undersize a
#     FIFO and rtlsim cannot.
#     """
#     if len(dims) == 0 or any(x < 1 for x in dims) or fm_size < 1:
#         return None
#     _, _, fp_inc, _ = nest_params(dims, coefs, fm_size)
#     level = outer_level(dims)
#     _, freed, closing = block_pattern(dims, coefs, fp_inc, level)
#     if (dims[level] - 1) * int(freed.sum()) + int(closing.sum()) != fm_size:
#         return None  # the free pointer does not hand back a frame of slots
#     settled = frame_blocks(dims, coefs, fm_size)
#     if settled is None:
#         return None
#     inner = list(dims[outer_level(dims) + 1 :])
#     return Characteristic_Node(
#         name,
#         [(count, _fold(block_delta(block), inner, 0, name)) for count, block in settled[0]],
#         False,
#     )
