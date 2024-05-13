# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

import pytest
import os
import numpy as np
import copy
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def make_single_dwc_modelwrapper(in_shape, out_shape, inWidth, outWidth, cropping, padding, finn_dtype, impl_style):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    optype = "StreamingDataWidthConverter"

    DWC_node = helper.make_node(
        optype,
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        in_shape=in_shape,
        out_shape=out_shape,
        inWidth=inWidth,
        outWidth=outWidth,
        cropping=cropping,
        padding=padding,
        preferred_impl_style=impl_style,

        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(nodes=[DWC_node], name="dwc_graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="dwc-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}




@pytest.mark.parametrize(
    "config",
    [
        # the LCM <= shape[1] hard constraint should hold
        # for the values after padding/cropping
         ([1, 2, 8],[1, 2, 10], 4, 6, 0, 1, DataType["INT2"]), # padding output so its actually 4->4
        #this testcase fails rtlsim
        ([1, 36], [1, 24], 8, 6, 1, 0, DataType["INT2"]), # cropped input so its actually 6->6
        # input and output shapes are provided with the padding and cropping
        # passing testcases
        ([1, 2, 8], [1, 2, 8], 4, 4, 0, 0, DataType["INT2"]),
        ([1, 8], [1, 8], 4, 4, 0, 0, DataType["INT2"]),
       
        ([1, 2, 512],[1, 2, 513], 1024, 1026, 0, 1, DataType["INT2"]),



      #  ([1, 2, 12], 8, 14, 1, 1,  DataType["INT2"]),
      #  ([1, 2, 8], 8, 18, 1,  DataType["INT2"]),
      #  ([1, 2, 24], 4, 6, 0, DataType["INT2"]),
      #  ([1, 24], 6, 4, 0, DataType["INT2"]),
        
      #  ([1, 5, 512], 1024, 1028, 2, DataType["INT2"]),
       
       # ([1, 2, 512], 1026, 1024, 1, 0, DataType["INT2"]),
       # ([1, 96], 28, 28, 2, 2, DataType["INT2"]),
        
    ],
)
@pytest.mark.parametrize("exec_mode", ["cppsim","rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
# impl style
@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.vivado
def test_fpgadataflow_dwc(config, exec_mode, impl_style):
    in_shape, out_shape, inWidth, outWidth, cropping, padding, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, in_shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(in_shape, out_shape, inWidth, outWidth,
                                          cropping, padding, finn_dtype, impl_style)
    # verify abstraction level execution
    y = oxe.execute_onnx(model, input_dict)["outp"]
    golden_shape = copy.copy(out_shape)

    y_golden = np.zeros((out_shape))

    

   # adjusting the output shape if padding has been introduced
   # out_els = outWidth / finn_dtype.bitwidth() - padding
   # num_words = int(shape[-1] // out_els) 
   # golden_shape[-1] += padding * num_words



    assert y.shape == tuple(golden_shape), """The output shape is incorrect."""

    model = model.transform(SpecializeLayers())
    model = model.transform(GiveUniqueNodeNames())
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(PrepareIP(test_fpga_part, 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareRTLSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]


    assert y.shape == tuple(golden_shape), """The output shape is incorrect."""
    #Delete previous run results if exist



@pytest.mark.parametrize(
    "config",
    [

        # stitching currently fails tests with non-zero resizing
        # RTL outputs are correct thus the issue is in the 
        # way rtsim_exec takes folded_output_shape()
        # TODO: adjust these public functions to work for
        # rtlsim_exec 
        # the LCM <= shape[1] hard constraint should hold
        # for the values after padding/cropping
        ([1, 2, 8], 4, 4, 0, 0, DataType["INT2"]),
        ([1, 8], 4, 4, 0, 0, DataType["INT2"]),
        ([1, 2, 8], 4, 6, 0, 1, DataType["INT2"]), # padding output so its actually 4->4
        ([1, 24], 6, 6, 1, 0, DataType["INT2"]), # cropped input so its actually 4->6
        ([1, 2, 512], 1024, 1026, 0, 1, DataType["INT2"]),
      #  ([1, 2, 12], 8, 14, 1, 1,  DataType["INT2"]),
      #  ([1, 2, 8], 8, 18, 1,  DataType["INT2"]),
      #  ([1, 2, 24], 4, 6, 0, DataType["INT2"]),
      #  ([1, 24], 6, 4, 0, DataType["INT2"]),
        
      #  ([1, 5, 512], 1024, 1028, 2, DataType["INT2"]),
       
       # ([1, 2, 512], 1026, 1024, 1, 0, DataType["INT2"]),
       # ([1, 96], 28, 28, 2, 2, DataType["INT2"]),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.slow
# impl style
@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.vivado
def test_fpgadataflow_dwc_stitched_rtlsim(config, impl_style):
    shape, inWidth, outWidth, cropping, padding, finn_dtype = config

    golden_shape = copy.copy(shape)

    # adjusting the output shape if padding has been introduced
    out_els = outWidth / finn_dtype.bitwidth() - padding
    num_words = int(shape[-1] // out_els) 
    golden_shape[-1] += padding * num_words

    test_fpga_part = "xc7z020clg400-1"
    target_clk_ns = 10.0
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(shape, inWidth, outWidth, 
                                         cropping, padding, finn_dtype, impl_style)
    model = model.transform(SpecializeLayers())
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")
    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert y.shape == tuple(golden_shape), """The output shape is incorrect."""

