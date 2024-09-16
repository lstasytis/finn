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

import numpy as np
import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

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

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for ConvInpGen."
        # implement tensor with correct shape
        return super().make_const_shape_op(oshape)

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

    def verify_node(self):
        pass

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

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        num_output_elems = np.prod(folded_oshape[:-1])
        return num_output_elems

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

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_im2col = ModelWrapper(qonnx_make_model(graph_im2col, **onnx_kwargs))
        model_im2col.set_tensor_datatype(node.input[0], self.get_input_datatype())
        # use execution function from Im2Col node
        # this automatically updates the execution context
        inst = getCustomOp(im2col_node)
        inst.execute_node(context, model_im2col.graph)


    def prepare_kwargs_for_characteristic_fx(self):

        # key parameters
        IFMDim_x = self.get_nodeattr("IFMDim")[0]
        OFMDim_x = self.get_nodeattr("OFMDim")[0]
        ConvKernelDim_x = self.get_nodeattr("ConvKernelDim")[0]
        Stride_x = self.get_nodeattr("Stride")[0]

        IFMDim_y = self.get_nodeattr("IFMDim")[1]
        OFMDim_y = self.get_nodeattr("OFMDim")[1]
        ConvKernelDim_y = self.get_nodeattr("ConvKernelDim")[1]
        Stride_y = self.get_nodeattr("Stride")[1]

        SIMD = self.get_nodeattr("SIMD")
        
        IFMChannels = self.get_nodeattr("IFMChannels")
        
        
        dilation = self.get_nodeattr("Dilation")
        DEPTHWISE = self.get_nodeattr("depthwise")
        parallel_window = self.get_nodeattr("parallel_window")
        is1d = self.get_nodeattr("is1D")
       # m = self.get_nodeattr("m")
       # flip = self.get_nodeattr("flip")

        SIMD_COUNT  = int(IFMChannels / SIMD)
        OUTPUT_SIZE = OFMDim_x * ConvKernelDim_x * SIMD_COUNT
        INPUT_SIZE = IFMDim_x * SIMD_COUNT
        WINDOW_SIZE = ConvKernelDim_x * SIMD_COUNT
        if DEPTHWISE:
            BUFFER_SIZE = ConvKernelDim_x * SIMD_COUNT
            READ_CYCLES = SIMD_COUNT * (ConvKernelDim_x-1) - (ConvKernelDim_x-1)
            FINISH = IFMDim_x-ConvKernelDim_x-2
        else:
            BUFFER_SIZE = (ConvKernelDim_x-1) * SIMD_COUNT
            READ_CYCLES = 0
            FINISH = 0

        OCNT_INITIAL = BUFFER_SIZE + (Stride_x - 1)

        DEFAULT_FIFO_DEPTH = 2

        multiplying_factor = int(IFMChannels/SIMD)
        number_blocks = int(ConvKernelDim_y/Stride_y + 1) 
        cycles_write_block = OFMDim_x * ConvKernelDim_x * ConvKernelDim_y * multiplying_factor
        cycles_read_block = Stride_x * IFMDim_x * multiplying_factor
        max_cycles = max(cycles_write_block,cycles_read_block)
        baseIter = IFMDim_x * ConvKernelDim_y * multiplying_factor + OFMDim_y * max(cycles_write_block,cycles_read_block)
        initial_buffer = IFMDim_x * ConvKernelDim_y *multiplying_factor

        READ_DELAY = number_blocks * ConvKernelDim_x*ConvKernelDim_y*OFMDim_x*OFMDim_y*multiplying_factor - ConvKernelDim_x*ConvKernelDim_y*OFMDim_x
        READ_ITES = int((baseIter-OFMDim_y) / max(cycles_write_block,cycles_read_block))

       # assert True == False
        kwargs = (SIMD_COUNT,Stride_x,Stride_y,OUTPUT_SIZE,INPUT_SIZE,
            WINDOW_SIZE,BUFFER_SIZE,READ_CYCLES,OCNT_INITIAL,
            DEPTHWISE,DEFAULT_FIFO_DEPTH, is1d,
            multiplying_factor,number_blocks,cycles_write_block,
            cycles_read_block,max_cycles,baseIter,initial_buffer,
            FINISH,OFMDim_y,READ_DELAY,READ_ITES
            )


       # assert True==False

        return kwargs

    def characteristic_fx_input(self, txns, cycles, counter, kwargs):
        # Compute one period of the input characteristic function

        (SIMD_COUNT,Stride_x,Stride_y,OUTPUT_SIZE,INPUT_SIZE,
         WINDOW_SIZE,BUFFER_SIZE,READ_CYCLES,
         OCNT_INITIAL, DEPTHWISE,DEFAULT_FIFO_DEPTH,is1d,
                     multiplying_factor,number_blocks,cycles_write_block,
            cycles_read_block,max_cycles,baseIter,initial_buffer,FINISH,OFMDim_y,READ_DELAY,
            READ_ITES) = kwargs

        
        if DEPTHWISE:
            OCNT_MAX = BUFFER_SIZE
            ocnt = SIMD_COUNT

        else:
            OCNT_MAX = WINDOW_SIZE
            if OCNT_INITIAL < WINDOW_SIZE:
                ocnt = OCNT_INITIAL
            else: ocnt=-1
        

        # fifo filling
        for i in range(0,DEFAULT_FIFO_DEPTH):
            txns.append(counter)
            counter+=1
            cycles+=1


        # main function
            
        inp_count = 0

        if is1d:
            for i in range(0,OUTPUT_SIZE):
                txns.append(counter)
                we = (i < OCNT_MAX) or (ocnt < (SIMD_COUNT * Stride_x))
                re = i > 0

                if re:
                    ocnt+=1
                    if ocnt == OCNT_MAX:
                        ocnt = 0
                if we:
                    if inp_count < INPUT_SIZE-DEFAULT_FIFO_DEPTH:
                        counter+=1
                        inp_count+=1
                
                cycles+=1
        else:

            for i in range(0,initial_buffer+cycles_read_block-1):
                txns.append(counter)
                cycles+=1   
                counter+=1

            txns.append(counter)
            cycles+=1 # one  extra for loop tail

            for i in range(0,OFMDim_y-1):
                for j in range(0,cycles_write_block-cycles_read_block):
                    txns.append(counter)
                    cycles+=1
                                       

                for j in range(0,cycles_read_block-1):
                    if i < OFMDim_y-2:
                        counter+=1
                        txns.append(counter) 
                        cycles+=1
                 #   else:
                    #   if j < FINISH:
                    #        counter+=1
                    #        txns.append(counter) 
                     #       cycles+=1
#
        return txns, cycles, counter

    def characteristic_fx_output(self, txns, cycles, counter, kwargs):
        # Compute one period of the output characteristic function

        (SIMD_COUNT,Stride_x,Stride_y,OUTPUT_SIZE,INPUT_SIZE,
         WINDOW_SIZE,BUFFER_SIZE,READ_CYCLES,
         OCNT_INITIAL, DEPTHWISE,DEFAULT_FIFO_DEPTH, is1d,
                     multiplying_factor,number_blocks,cycles_write_block,
            cycles_read_block,max_cycles,baseIter,initial_buffer,FINISH,OFMDim_y,READ_DELAY,
            READ_ITES) = kwargs

        # HYPER PARAMETERS
        


        INITIAL_LOOP_CYCLES = 5


        if is1d:
            for i in range(0,INITIAL_LOOP_CYCLES):
                txns.append(counter)
                cycles+=1   

            for i in range(0,READ_CYCLES):
                txns.append(counter)
                cycles+=1   



            for i in range(0,OUTPUT_SIZE):
                txns.append(counter)
                counter+=1
                cycles+=1
        else:
 
            for i in range(0,initial_buffer+INITIAL_LOOP_CYCLES-1):
                txns.append(counter)
                cycles+=1  

            for i in range(0,baseIter-initial_buffer):
                txns.append(counter)
                counter+=1
                cycles+=1            

        return txns, cycles, counter


    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out": []},
        }

        ignore = self.get_nodeattr("ipgen_ignore")
        if ignore == 0: # this node is being derived using RTLSIM
            # RTL-based flow
            super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)
            return
        

        # Analytical flow 
        
        txns_in = {key: [] for (key, value) in io_dict["inputs"].items() if "in" in key}
        txns_out = {key: [] for (key, value) in io_dict["outputs"].items() if "out" in key}

        all_txns_in = np.empty((len(txns_in.keys()), 2 * period), dtype=np.int32)
        all_txns_out = np.empty((len(txns_out.keys()), 2 * period), dtype=np.int32)


        self.set_nodeattr("io_chrc_period",period)




        txn_in = []
        txn_out = []

        # INPUT

        counter = 0
        padding = 0
        

        kwargs = self.prepare_kwargs_for_characteristic_fx()

        
        # first period
        cycles = 0
        txn_in, cycles, counter = self.characteristic_fx_input(txn_in,cycles,counter,kwargs)

        txn_in += [counter] * (period-cycles)
        padding+=(period*-cycles)
        

        # second period
        cycles = period
        txn_in, cycles, counter = self.characteristic_fx_input(txn_in,cycles,counter,kwargs)


        txn_in += [counter] * (period*2-cycles)
        padding+=(period*2-cycles)

        # final assignments
        all_txns_in[0, :] = np.array(txn_in)
        self.set_nodeattr("io_chrc_in", all_txns_in)
        self.set_nodeattr("io_chrc_pads_in", padding)


        # OUTPUT
        
        counter = 0
        cycles = 0  
        padding = 0          


        txn_out, cycles, counter = self.characteristic_fx_output(txn_out,cycles,counter,kwargs)


        txn_out += [counter] * (period-cycles)
        padding += (period*-cycles)

        cycles = period

        txn_out, cycles, counter = self.characteristic_fx_output(txn_out,cycles,counter,kwargs)

        txn_out += [counter] * (period*2-cycles)
        padding+=(period*2-cycles)


        all_txns_out[0, :] = np.array(txn_out)   
        self.set_nodeattr("io_chrc_out", all_txns_out)
        self.set_nodeattr("io_chrc_pads_out", padding)
