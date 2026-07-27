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

import math
import numpy as np
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

    def get_tree_model_uniform_distribution_based(self):
        def distribute_outputs_uniform(
            out_total, in_total, stride_y=1, stride_x=1, feature_map_x=1, kernel_x=1, kernel_y=1
        ):
            if in_total == 0:
                return [out_total]

            # if kernel_y > 1:
            # stride_y = stride_y - (kernel_y-1) // 2
            # if kernel_x > 1:
            # stride_x = stride_x - (kernel_x-1) // 2

            spacing_y = max(feature_map_x * (stride_y - 1), 1)
            spacing_x = max((stride_x - 1 + (kernel_x - 1) // 2), 1)

            weights = []
            for i in range(in_total):
                weight = 1
                if stride_y > 1:
                    if i % spacing_y == 0:
                        weight += spacing_y
                if stride_x > 1:
                    if i % spacing_x == 0:
                        weight += spacing_x
                weights.append(weight)

            # Normalize weights to match out_total
            total_weight = sum(weights)
            raw_counts = [w * out_total / total_weight for w in weights]

            # Round to nearest integers
            int_counts = [int(round(x)) for x in raw_counts]

            # Adjust rounding error
            diff = sum(int_counts) - out_total
            if diff != 0:
                adjustments = sorted(
                    enumerate(raw_counts), key=lambda x: x[1] - int_counts[x[0]], reverse=(diff > 0)
                )
                for i, _ in adjustments:
                    if diff == 0:
                        break
                    int_counts[i] -= int(diff / abs(diff))
                    diff -= int(diff / abs(diff))

            return int_counts

        IMPL_STYLE = "rtl" if "_rtl" in (self.__class__.__name__) else "hls"
        assert IMPL_STYLE in ["rtl", "hls"], "Implementation style must be 'rtl' or 'hls'"

        # Extract node attributes
        ifm_dim_y, ifm_dim_x = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        simd = self.get_nodeattr("SIMD")
        k_h, k_w = self.get_nodeattr("ConvKernelDim")
        stride_y, stride_x = self.get_nodeattr("Stride")
        dilation_y, dilation_x = self.get_nodeattr("Dilation")
        is1d = self.get_nodeattr("is1D")
        parallel_window = self.get_nodeattr("parallel_window")
        # numReps = 1

        assert ifm_ch % simd == 0
        factor = ifm_ch // simd
        ofm_dim_y = compute_conv_output_dim(ifm_dim_y, k_h, stride_y, 0, dilation_y)
        ofm_dim_x = compute_conv_output_dim(ifm_dim_x, k_w, stride_x, 0, dilation_x)
        total_outputs = ofm_dim_y * ofm_dim_x
        total_inputs = ifm_dim_y * ifm_dim_x
        if parallel_window:
            k_h = 1
            k_w = 1
        # if not is1d:
        #     # 2D convolution
        #     output_tokens = total_outputs * (k_h * k_w)
        # else:
        #     # 1D convolution
        #     output_tokens = total_outputs * (k_h)

        # key parameters
        # IFMDim_x = self.get_nodeattr("IFMDim")[0]
        # OFMDim_x = self.get_nodeattr("OFMDim")[0]
        ConvKernelDim_x = self.get_nodeattr("ConvKernelDim")[0]
        # Stride_x = self.get_nodeattr("Stride")[0]

        # OFMDim_y = self.get_nodeattr("OFMDim")[1]
        ConvKernelDim_y = self.get_nodeattr("ConvKernelDim")[1]
        # Stride_y = self.get_nodeattr("Stride")[1]

        # SIMD = self.get_nodeattr("SIMD")

        # IFMChannels = self.get_nodeattr("IFMChannels")

        DEPTHWISE = self.get_nodeattr("depthwise")
        is1d = self.get_nodeattr("is1D")

        # SF = IFMChannels // SIMD
        # OUTPUT_SIZE = OFMDim_x * ConvKernelDim_x * SF
        # INPUT_SIZE = IFMDim_x * SF
        # WINDOW_SIZE = ConvKernelDim_x * SF
        # if DEPTHWISE:
        #     BUFFER_SIZE = ConvKernelDim_x * SF
        #     READ_CYCLES = SF * (ConvKernelDim_x - 1) - (ConvKernelDim_x - 1)
        #     FINISH = IFMDim_x - ConvKernelDim_x - 2
        # else:
        #     BUFFER_SIZE = (ConvKernelDim_x - 1) * SF
        #     READ_CYCLES = 0
        #     FINISH = 0

        assert ifm_ch % simd == 0
        factor = ifm_ch // simd

        # OCNT_INITIAL = BUFFER_SIZE + (Stride_x - 1)

        # DEFAULT_FIFO_DEPTH = 2

        ofm_dim_y = compute_conv_output_dim(ifm_dim_y, k_h, stride_y, 0, dilation_y)
        ofm_dim_x = compute_conv_output_dim(ifm_dim_x, k_w, stride_x, 0, dilation_x)

        if DEPTHWISE:
            ofm_dim_y = ofm_dim_y * ConvKernelDim_y
            ofm_dim_x = ofm_dim_x * ConvKernelDim_x

        if DEPTHWISE:
            flip_factor = factor
        else:
            flip_factor = 1

        total_outputs = ofm_dim_y * ofm_dim_x * flip_factor
        total_inputs = ifm_dim_y * ifm_dim_x * flip_factor
        if parallel_window:
            k_h = 1
            k_w = 1
        # if not is1d:
        #     # 2D convolution
        #     output_tokens = total_outputs * (k_h * k_w)
        # else:
        #     # 1D convolution
        #     output_tokens = total_outputs * (k_h)

        ch_write = Characteristic_Node("Output Write", [(factor // flip_factor, [0, 1])], True)
        ch_read = Characteristic_Node("Streamed Read", [(factor // flip_factor, [1, 0])], True)
        ch_both = Characteristic_Node("Streamed Read", [(factor // flip_factor, [1, 1])], True)

        out_total = np.prod(self.get_folded_output_shape()[:-1]) // factor * flip_factor
        in_total = np.prod(self.get_folded_input_shape()[:-1]) // factor * flip_factor

        # Calculate startup and steady reads
        if not is1d:
            startup_reads = (k_h - 1) * ifm_dim_x + k_w  # - (ifm_dim_x-k_w)
            #  startup_writes = ofm_dim_x - (ofm_dim_x-k_w) // (stride_x * stride_y)# *
            # factor # we can only write the middle in this section!!!
            if not DEPTHWISE:
                if k_h > 1:
                    startup_writes = ofm_dim_x  # k_w*stride_x # // (stride_x)
                else:
                    startup_writes = ofm_dim_x  # // (stride_x * stride_y)
            else:
                if k_h > 1:
                    startup_writes = 0
                else:
                    startup_writes = 0
        else:
            startup_reads = ifm_dim_x
            startup_writes = ofm_dim_x // stride_x

        startup_reads = startup_reads * flip_factor
        startup_writes = startup_writes * flip_factor

        # startup_reads = 0
        steady_reads = total_inputs - startup_reads
        steady_writes = total_outputs - startup_writes

        total_inputs = total_inputs - startup_reads
        total_outputs = total_outputs - startup_writes
        # inputs_read = startup_reads

        if startup_writes == 0:
            offset_writing = 1
        else:
            offset_writing = 0

        # Steady-state reads > 0, normal case
        # Spread steady reads evenly across output_tokens cycles
        in_total = in_total - startup_reads
        out_total = out_total - startup_writes

        if startup_writes > startup_reads:
            schedule = distribute_outputs_uniform(
                startup_writes, startup_reads, stride_x, stride_y, k_w, k_h, ifm_dim_x
            )
            per_cycle_nodes = []

            for tokens_this_cycle in schedule:
                cycle = Characteristic_Node(
                    "Cycle",
                    [
                        (1 - offset_writing, ch_both),
                        (
                            1,
                            Characteristic_Node(
                                "Output Write",
                                [(tokens_this_cycle - 1 + offset_writing, ch_write)],
                                False,
                            ),
                        ),
                    ],
                    False,
                )
                per_cycle_nodes.append((1, cycle))

            startup = Characteristic_Node("Processing Loop", per_cycle_nodes, False)
        else:
            schedule = distribute_outputs_uniform(
                startup_reads, startup_writes, stride_x, stride_y, k_w, k_h, ifm_dim_x
            )
            per_cycle_nodes = []

            for tokens_this_cycle in schedule:
                cycle = Characteristic_Node(
                    "Cycle",
                    [
                        (1 - offset_writing, ch_both),
                        (
                            1,
                            Characteristic_Node(
                                "Input Read",
                                [(tokens_this_cycle - 1 + offset_writing, ch_read)],
                                False,
                            ),
                        ),
                    ],
                    False,
                )
                per_cycle_nodes.append((1, cycle))

            startup = Characteristic_Node("Processing Loop", per_cycle_nodes, False)

        if out_total > in_total:
            if steady_reads <= 0:
                return Characteristic_Node(
                    "SlidingWindow_2D", [(1, startup), (steady_writes, ch_write)], False
                )

            schedule = distribute_outputs_uniform(
                out_total, in_total, stride_x, stride_y, k_w, k_h, ifm_dim_x
            )
            per_cycle_nodes = []

            for tokens_this_cycle in schedule:
                cycle = Characteristic_Node(
                    "Cycle",
                    [
                        (1, ch_both),
                        (
                            1,
                            Characteristic_Node(
                                "Output Write", [(tokens_this_cycle - 1, ch_write)], False
                            ),
                        ),
                    ],
                    False,
                )
                per_cycle_nodes.append((1, cycle))

            steady = Characteristic_Node("Processing Loop", per_cycle_nodes, False)

            return Characteristic_Node("SlidingWindow_2D", [(1, startup), (1, steady)], False)

        else:
            if steady_reads <= 0:
                return Characteristic_Node(
                    "SlidingWindow_2D", [(1, startup), (steady_writes, ch_write)], False
                )

            schedule = distribute_outputs_uniform(
                in_total, out_total, stride_x, stride_y, k_w, k_h, ifm_dim_x
            )
            per_cycle_nodes = []

            for tokens_this_cycle in schedule:
                cycle = Characteristic_Node(
                    "Cycle",
                    [
                        (1, ch_both),
                        (
                            1,
                            Characteristic_Node(
                                "Output Write", [(tokens_this_cycle - 1, ch_read)], False
                            ),
                        ),
                    ],
                    False,
                )
                per_cycle_nodes.append((1, cycle))

            steady = Characteristic_Node("Processing Loop", per_cycle_nodes, False)

            return Characteristic_Node("SlidingWindow_2D", [(1, startup), (1, steady)], False)

    def get_tree_model(self):
        ifm_dim_y, ifm_dim_x = self.get_nodeattr("IFMDim")
        ifm_ch = self.get_nodeattr("IFMChannels")
        simd = self.get_nodeattr("SIMD")
        k_y, k_x = self.get_nodeattr("ConvKernelDim")
        stride_y, stride_x = self.get_nodeattr("Stride")
        dilation_y, dilation_x = self.get_nodeattr("Dilation")
        parallel_window = self.get_nodeattr("parallel_window")
        depthwise = self.get_nodeattr("depthwise")
        SF = ifm_ch // simd

        def mkleaf(name, phases):
            return Characteristic_Node(name, [(cnt, vals) for cnt, vals in phases if cnt > 0], True)

        # 1x1 pass-through / depthwise-equivalent
        if k_y == 1 and k_x == 1:
            n_tok = SF * ifm_dim_y * ifm_dim_x
            return Characteristic_Node(
                "k1_pass",
                [(1, [0, 1]), (1, [1, 0]), (n_tok - 1, [1, 1])],
                True,
            )

        # depthwise, default impl, k=2 s=2, channel_factor > 1
        if (
            parallel_window == 0
            and depthwise == 1
            and simd != ifm_ch
            and k_y == 2
            and k_x == 2
            and stride_y == 2
            and stride_x == 2
            and dilation_y == 1
            and dilation_x == 1
        ):
            ofm_dim_y = math.floor((ifm_dim_y - k_y) / stride_y) + 1
            n = ifm_dim_x * SF
            pair_cnt = (SF - 2) // 2
            w = ifm_dim_x
            c = SF + 2 - (w - 2) * pair_cnt

            prefix = mkleaf(
                "dw_prefix",
                [
                    (1, [0, 1]),
                    (2, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (1, [1, 1]),
                    (n - SF - 1, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (n - (w - 2) * pair_cnt, [1, 1]),
                ],
            )

            a_base = mkleaf(
                "dw_row_start_base",
                [
                    (1, [0, 1]),
                    (1, [1, 1]),
                    (3, [0, 1]),
                    (w, [1, 1]),
                ],
            )
            a_pair = mkleaf(
                "dw_row_start_pair",
                [
                    (2, [0, 1]),
                    (1, [1, 1]),
                    (3, [0, 1]),
                    (1, [1, 1]),
                    (3, [0, 1]),
                    (w, [1, 1]),
                ],
            )
            tail_mid_both = 2 * n - (1 + w) - pair_cnt * (w + 2) - (4 * SF - 3)
            tail_mid = mkleaf(
                "dw_row_tail_mid",
                [
                    (3 * SF - 3, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (tail_mid_both, [1, 1]),
                ],
            )
            tail_last = mkleaf(
                "dw_row_tail_last",
                [
                    (3 * SF - 3, [1, 0]),
                    (1, [1, 1]),
                    (SF - 1, [1, 0]),
                    (tail_mid_both - c, [1, 1]),
                    (n + 2 * pair_cnt, [0, 1]),
                ],
            )

            middle = Characteristic_Node(
                "dw_middle_row",
                [
                    (1, a_base),
                    (pair_cnt, a_pair),
                    (1, tail_mid),
                ],
                False,
            )
            last = Characteristic_Node(
                "dw_last_row",
                [
                    (1, a_base),
                    (pair_cnt, a_pair),
                    (1, tail_last),
                ],
                False,
            )

            return Characteristic_Node(
                "dw_pw0_k2s2",
                [
                    (1, prefix),
                    (ofm_dim_y - 2, middle),
                    (1, last),
                ],
                False,
            )

        # Special stride-3 pw=1 non-depthwise branch.
        if (
            parallel_window == 1
            and depthwise == 0
            and k_y == 2
            and k_x == 2
            and stride_y == 3
            and stride_x == 3
            and dilation_y == 1
            and dilation_x == 1
            and SF == 1
        ):
            ofm_dim_y = math.floor((ifm_dim_y - k_y) / stride_y) + 1
            ofm_dim_x = math.floor((ifm_dim_x - k_x) / stride_x) + 1
            first_valid_read = (k_y - 1) * ifm_dim_x + (k_x - 1) + 1
            prefix = first_valid_read
            gap_x = stride_x - 1
            row_gap = stride_y * ifm_dim_x - (ofm_dim_x - 1) * stride_x - 1

            ch_both = mkleaf("both", [(1, [1, 1])])
            ch_read = mkleaf("read", [(1, [1, 0])])
            step = Characteristic_Node("step", [(1, ch_both), (gap_x, ch_read)], False)
            full_row = Characteristic_Node(
                "full_row",
                [(ofm_dim_x - 1, step), (1, ch_both), (row_gap, ch_read)],
                False,
            )
            last_row = Characteristic_Node(
                "last_row",
                [(ofm_dim_x - 1, step)],
                False,
            )
            return Characteristic_Node(
                "pw1_k2_s3",
                [(1, ch_both), (prefix, ch_read), (ofm_dim_y - 1, full_row), (1, last_row)],
                False,
            )

        # Generic parallel_window=1 model for remaining multi-tap cases.
        if parallel_window == 1:
            eff_k_y = k_y + (k_y - 1) * (dilation_y - 1)
            eff_k_x = k_x + (k_x - 1) * (dilation_x - 1)
            ofm_dim_y = math.floor((ifm_dim_y - eff_k_y) / stride_y) + 1
            ofm_dim_x = math.floor((ifm_dim_x - eff_k_x) / stride_x) + 1

            start_y = eff_k_y - 1
            start_x = eff_k_x - 1
            top_zero = start_y * ifm_dim_x * SF
            left_zero = start_x * SF
            burst = SF
            gap_x = (stride_x - 1) * SF
            last_valid_x = start_x + (ofm_dim_x - 1) * stride_x
            tail_x = (ifm_dim_x - 1 - last_valid_x) * SF
            between_rows_zero = (stride_y - 1) * ifm_dim_x * SF
            trailing_rows_zero = (ifm_dim_y - 1 - (start_y + (ofm_dim_y - 1) * stride_y)) * ifm_dim_x * SF
            carry_write = 1 if (tail_x == 0 and trailing_rows_zero == 0) else 0

            def build_row(trim_last):
                phases = []
                if left_zero > 0:
                    phases.append((left_zero, [1, 0]))
                for ox in range(ofm_dim_x):
                    burst_cnt = burst
                    if trim_last and trailing_rows_zero == 0 and tail_x == 0 and ox == ofm_dim_x - 1:
                        burst_cnt -= 1
                    if burst_cnt > 0:
                        phases.append((burst_cnt, [1, 1]))
                    if ox < ofm_dim_x - 1 and gap_x > 0:
                        phases.append((gap_x, [1, 0]))
                tail_cnt = tail_x
                if trim_last and trailing_rows_zero == 0 and tail_x > 0:
                    tail_cnt -= 1
                if tail_cnt > 0:
                    phases.append((tail_cnt, [1, 0]))
                return mkleaf("row", phases)

            row_full = build_row(False)
            row_short = build_row(True)
            top_zero_leaf = mkleaf("top_zero", [(top_zero, [1, 0])])
            between_rows_leaf = mkleaf("between_rows", [(between_rows_zero, [1, 0])])
            trailing_rows_leaf = mkleaf("trailing_rows", [(trailing_rows_zero - 1, [1, 0])])
            carry_leaf = mkleaf("carry", [(1, [0, carry_write])])
            bubble_leaf = mkleaf("bubble", [(1, [1, 0])])

            common_row_parts = [(1, row_full)]
            if between_rows_zero > 0:
                common_row_parts.append((1, between_rows_leaf))
            common_row = Characteristic_Node("common_row", common_row_parts, False)

            if trailing_rows_zero > 0:
                last_part_children = [(1, row_full)]
                if trailing_rows_zero - 1 > 0:
                    last_part_children.append((1, trailing_rows_leaf))
                last_part = Characteristic_Node("last_part", last_part_children, False)
            else:
                last_part = row_short

            scan_children = []
            if top_zero > 0:
                scan_children.append((1, top_zero_leaf))
            if ofm_dim_y > 1:
                scan_children.append((ofm_dim_y - 1, common_row))
            scan_children.append((1, last_part))
            scan_prefix = Characteristic_Node("scan_prefix", scan_children, False)

            return Characteristic_Node(
                "pw1_generic",
                [(1, carry_leaf), (1, bubble_leaf), (1, scan_prefix)],
                False,
            )

        # k=[2,2] pw=0 stride=[2,2] dw=0
        if k_y == 2 and k_x == 2 and parallel_window == 0 and depthwise == 0 and stride_y == 2 and stride_x == 2:
            n_pix = ifm_dim_y * ifm_dim_x
            kernel_lines = math.ceil((ifm_dim_y - k_y + 1) / stride_y)

            main_both = SF * n_pix - 2 - 2 * SF - 4 * SF
            trailing_writes = SF * (kernel_lines - 1) + 1

            swg = Characteristic_Node(
                "k=2x2 pw=0 s=2x2",
                [
                    (1, [0, 1]),
                    (2, [1, 0]),
                    (2 * SF, [1, 1]),
                    (4 * SF, [1, 0]),
                    (main_both, [1, 1]),
                    (trailing_writes, [0, 1]),
                ],
                True,
            )
            return swg

        # Baseline branch for remaining pw=0 / other special cases
        stride_y_skips = (stride_y - 1) * ifm_dim_x

        kernels_in_line = math.ceil(
            (ifm_dim_x - (k_x - 1 + (k_x - 1) * (dilation_x - 1))) / stride_x
        )
        kernel_lines = math.ceil(
            (ifm_dim_y - ((k_y - 1) + (k_y - 1) * (dilation_y - 1))) / stride_y
        )

        shifts_x = (kernels_in_line - 1) * stride_x
        starting_index_x = k_x + (k_x - 1) * (dilation_x - 1)
        remainder_x = ifm_dim_x - (starting_index_x + shifts_x)

        shifts_y = (kernel_lines - 1) * stride_y
        starting_index_y = k_y + (k_y - 1) * (dilation_y - 1)
        remainder_y = (ifm_dim_y - (starting_index_y + shifts_y)) * ifm_dim_x

        reads_to_prepare_line = (k_x - 1) + (k_x - 1) * (dilation_x - 1)
        reads_to_prepare_first_line = ((k_y - 1) + (k_y - 1) * (dilation_y - 1)) * ifm_dim_x
        total_kernel_y = k_y + (k_y - 1) * (dilation_y - 1)
        first_line_kernel_buffer = k_x + (k_x - 1) * (dilation_x - 1)
        first_line_buffer = (total_kernel_y - 1) * ifm_dim_x

        if parallel_window == 1:
            writes_per_kernel = 1
        else:
            writes_per_kernel = k_y * k_x

        inner_line_buffer_reads = (stride_y - 1) * ifm_dim_x

        single_move_dif = writes_per_kernel - stride_x
        if single_move_dif > 0:
            do_both = stride_x
            writes_only = single_move_dif
            reads_only = 0
        else:
            do_both = writes_per_kernel
            reads_only = -single_move_dif
            writes_only = 0

        first_do_both = 0
        first_writes_only = writes_per_kernel
        first_reads_only = first_line_kernel_buffer

        absorbing_kernels = 0

        remaining_buffer_reads = inner_line_buffer_reads
        if inner_line_buffer_reads > 0 and ((kernels_in_line - 1) * writes_only) > 0:
            absorbing_kernels = min(
                math.floor((inner_line_buffer_reads) // writes_only), kernels_in_line - 1
            )
            absorbed_reads = absorbing_kernels * writes_only
            inner_line_buffer_reads -= absorbed_reads
            remaining_buffer_reads -= absorbed_reads

        first_reads = first_line_kernel_buffer + remaining_buffer_reads
        first_single_move_dif = writes_per_kernel - first_reads
        if first_single_move_dif > 0:
            first_do_both = first_reads
            first_writes_only = first_single_move_dif
            first_reads_only = 0
        else:
            first_do_both = writes_per_kernel
            first_reads_only = -first_single_move_dif
            first_writes_only = 0

        absolute_first_reads = first_line_kernel_buffer + first_line_buffer
        absolute_first_single_move_dif = writes_per_kernel - absolute_first_reads

        absolute_first_do_both = 0
        absolute_first_writes_only = writes_per_kernel
        absolute_first_reads_only = absolute_first_reads

        if depthwise == 0:
            if absolute_first_single_move_dif > 0:
                absolute_first_do_both = absolute_first_reads
                absolute_first_writes_only = absolute_first_single_move_dif
                absolute_first_reads_only = 0
            else:
                absolute_first_do_both = writes_per_kernel
                absolute_first_reads_only = -absolute_first_single_move_dif
                absolute_first_writes_only = 0

        ch_idle = Characteristic_Node("Output Write", [(SF, [0, 0])], True)
        ch_write = Characteristic_Node("Output Write", [(SF, [0, 1])], True)

        ch_read = Characteristic_Node("Streamed Read", [(SF, [1, 0])], True)
        ch_both = Characteristic_Node("Streamed Read+Write", [(SF, [1, 1])], True)

        if parallel_window == 2:
            ch_handle = Characteristic_Node("write out", [(1, ch_both)], False)

            handle_kernel = Characteristic_Node(
                "handle one kernel", [(1, ch_handle), (stride_x - 1, ch_read)], False
            )

            handle_last_kernel = Characteristic_Node(
                "handle last kernel",
                [
                    (1, ch_handle),
                    (remainder_x, ch_read),
                ],
                False,
            )

            handle_line = Characteristic_Node(
                "write_one_line",
                [
                    (reads_to_prepare_line, ch_read),
                    (kernels_in_line - 1, handle_kernel),
                    (1, handle_last_kernel),
                    (stride_y_skips, ch_read),
                ],
                False,
            )
            handle_last_line = Characteristic_Node(
                "write line without stride at end",
                [
                    (reads_to_prepare_line, ch_read),
                    (kernels_in_line, handle_kernel),
                    (remainder_y, ch_read),
                ],
                False,
            )
            swg = Characteristic_Node(
                "SlidingWindowGenerator",
                [
                    (1, ch_idle),
                    (reads_to_prepare_first_line, ch_read),
                    (kernel_lines - 1, handle_line),
                    (1, handle_last_line),
                ],
                False,
            )

        else:
            handle_absolute_kernel = Characteristic_Node(
                "handle one kernel",
                [
                    (absolute_first_do_both, ch_both),
                    (absolute_first_reads_only, ch_read),
                    (absolute_first_writes_only, ch_write),
                ],
                False,
            )

            handle_first_kernel = Characteristic_Node(
                "handle one kernel",
                [
                    (first_do_both, ch_both),
                    (first_reads_only, ch_read),
                    (first_writes_only, ch_write),
                ],
                False,
            )

            handle_kernel = Characteristic_Node(
                "handle one kernel",
                [
                    (do_both, ch_both),
                    (reads_only, ch_read),
                    (writes_only, ch_write),
                ],
                False,
            )

            handle_kernel_absorbed = Characteristic_Node(
                "handle one kernel with fused writes",
                [
                    (do_both + writes_only, ch_both),
                    (reads_only, ch_read),
                ],
                False,
            )

            handle_first_line = Characteristic_Node(
                "write first line",
                [
                    (1, handle_absolute_kernel),
                    (kernels_in_line - 1, handle_kernel),
                    (remainder_x, ch_read),
                ],
                False,
            )

            handle_line = Characteristic_Node(
                "write one inner line",
                [
                    (1, handle_first_kernel),
                    (absorbing_kernels, handle_kernel_absorbed),
                    (kernels_in_line - 1 - absorbing_kernels, handle_kernel),
                    (remainder_x, ch_read),
                ],
                False,
            )

            swg = Characteristic_Node(
                "SlidingWindowGenerator",
                [
                    (1, handle_first_line),
                    (kernel_lines - 1, handle_line),
                    (remainder_y, ch_read),
                ],
                False,
            )

        return swg

