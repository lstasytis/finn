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

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
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


def make_single_dwc_modelwrapper(shape, inWidth, outWidth, finn_dtype, impl_style, out_shape=None):
    out_shape = shape if out_shape is None else out_shape
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)

    optype = "StreamingDataWidthConverter"

    DWC_node = helper.make_node(
        optype,
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        in_shape=shape,
        out_shape=out_shape,
        inWidth=inWidth,
        outWidth=outWidth,
        dataType=str(finn_dtype.name),
        preferred_impl_style=impl_style,
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
    # config: (shape, inWidth, outWidth, dtype)
    "config",
    [
        # Non-integer ratio (RTL uses vpc genGeneric path)
        pytest.param(([1, 24], 24, 16, DataType["INT4"]), id="non-integer-ratio-down"),
        pytest.param(([1, 24], 16, 24, DataType["INT4"]), id="non-integer-ratio-up"),
        # Integer ratio - deserialization (RTL uses vpc genDes path)
        pytest.param(([1, 4], 2, 4, DataType["BIPOLAR"]), id="deserialize-2to4"),
        pytest.param(([1, 8], 24, 48, DataType["INT6"]), id="deserialize-24to48"),
        # Integer ratio - serialization (RTL uses vpc genSer path)
        pytest.param(([1, 4], 8, 4, DataType["UINT4"]), id="serialize-8to4"),
        pytest.param(([1, 4], 4, 2, DataType["INT2"]), id="serialize-4to2"),
        # Same width - wire-through (RTL uses vpc genWire path)
        pytest.param(([1, 2, 8], 16, 16, DataType["INT8"]), id="wire-through"),
        # Integer ratio - deserialization
        pytest.param(([1, 2, 8], 8, 16, DataType["INT2"]), id="deserialize-8to16"),
        # Wide stream spanning multiple spatial dimensions
        pytest.param(([1, 2, 8], 32, 16, DataType["INT2"]), id="wide-input-stream"),
        # lcm(128, 130) = 8320 > 8191, exercises HLS element-width splitting
        pytest.param(
            ([1, 65, 128], 128, 130, DataType["BINARY"]),
            id="oversized-lcm-hls-splitting",
        ),
    ],
)
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_dwc(config, exec_mode, impl_style):
    shape, inWidth, outWidth, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(shape, inWidth, outWidth, finn_dtype, impl_style)
    # verify abstraction level execution
    y = oxe.execute_onnx(model, input_dict)["outp"]
    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""
    assert y.shape == tuple(shape), """The output shape is incorrect."""

    model = model.transform(SpecializeLayers(test_fpga_part))
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

    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""
    assert y.shape == tuple(shape), """The output shape is incorrect."""


@pytest.mark.parametrize(
    "config",
    [
        ([1, 4], 2, 4, DataType["BIPOLAR"]),
        ([1, 4], 16, 8, DataType["UINT4"]),
        ([1, 2, 8], 16, 16, DataType["INT8"]),
        ([1, 2, 8], 4, 8, DataType["INT2"]),
    ],
)
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_dwc_stitched_rtlsim(config, impl_style):
    shape, inWidth, outWidth, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    target_clk_ns = 10.0
    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_dwc_modelwrapper(shape, inWidth, outWidth, finn_dtype, impl_style)
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    model.set_metadata_prop("exec_mode", "rtlsim")
    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert (
        y == x
    ).all(), """The output values are not the same as the
        input values anymore."""
    assert y.shape == tuple(shape), """The output shape is incorrect."""


# config: (in_shape, out_shape, inWidth, outWidth, dtype). The channel (last) dimension
# is zero-padded or cropped at the tail of every row. Only the RTL converter (vpc) pads
# or crops, within the last word of a row: what folding with channel padding produces.
DWC_PAD_CONFIGS = [
    # 10 channels produced 4/word (padded to 12), consumed 5/word
    pytest.param(([1, 2, 2, 12], [1, 2, 2, 10], 8, 10, DataType["INT2"]), id="crop-up"),
    pytest.param(([1, 2, 2, 10], [1, 2, 2, 12], 10, 8, DataType["INT2"]), id="pad-down"),
    pytest.param(([1, 4, 12], [1, 4, 15], 16, 20, DataType["INT4"]), id="pad-up"),
    pytest.param(([1, 2, 2, 10], [1, 2, 2, 8], 10, 4, DataType["BIPOLAR"]), id="crop-down"),
    pytest.param(([1, 2, 2, 24], [1, 2, 2, 26], 4, 13, DataType["BIPOLAR"]), id="pad-coprime"),
    pytest.param(([1, 2, 2, 30], [1, 2, 2, 32], 10, 8, DataType["INT2"]), id="pad-down-2"),
    pytest.param(([1, 3, 3, 12], [1, 3, 3, 15], 6, 5, DataType["BINARY"]), id="pad-multi-rep"),
    pytest.param(([1, 2, 2, 12], [1, 2, 2, 15], 8, 40, DataType["UINT8"]), id="pad-one-word"),
    # conv output into a padded fully connected layer
    pytest.param(([1, 1, 1, 12], [1, 15], 8, 10, DataType["INT2"]), id="pad-rank-change"),
]

# padding/cropping beyond the last word of a row, which no converter supports
DWC_PAD_UNSUPPORTED = [
    ([1, 2, 2, 8], [1, 2, 2, 16], 4, 4, DataType["BIPOLAR"]),
    ([1, 2, 2, 64], [1, 2, 2, 18], 16, 6, DataType["BIPOLAR"]),
    ([1, 2, 2, 30], [1, 2, 2, 40], 10, 8, DataType["INT2"]),
    ([1, 2, 64], [1, 2, 80], 8, 16, DataType["INT4"]),
]


@pytest.mark.parametrize("config", DWC_PAD_CONFIGS)
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("impl_style", ["hls", "rtl"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_dwc_padding(config, exec_mode, impl_style):
    in_shape, out_shape, inWidth, outWidth, finn_dtype = config

    test_fpga_part = "xc7z020clg400-1"
    x = gen_finn_dt_tensor(finn_dtype, in_shape)
    input_dict = prepare_inputs(x, finn_dtype)
    k = min(in_shape[-1], out_shape[-1])
    x_rows = x.reshape(-1, in_shape[-1])
    y_exp = np.zeros((x_rows.shape[0], out_shape[-1]), dtype=np.float32)
    y_exp[:, :k] = x_rows[:, :k]
    y_exp = y_exp.reshape(out_shape)

    model = make_single_dwc_modelwrapper(
        in_shape, inWidth, outWidth, finn_dtype, impl_style, out_shape=out_shape
    )
    assert getCustomOp(model.graph.node[0]).rtl_supported()
    y = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y == y_exp).all(), "Abstraction-level padding/cropping is incorrect."

    # padding/cropping is RTL-only, whatever the preferred style
    model = model.transform(SpecializeLayers(test_fpga_part))
    assert model.graph.node[0].op_type == "StreamingDataWidthConverter_rtl"
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

    assert y.shape == tuple(out_shape), "The output shape is incorrect."
    # zero bits decode to -1 for BIPOLAR, so only the carried elements are compared there
    if finn_dtype == DataType["BIPOLAR"]:
        assert (y[..., :k] == y_exp[..., :k]).all(), "Carried elements are incorrect."
    else:
        assert (y == y_exp).all(), "Padded/cropped output is incorrect."


@pytest.mark.parametrize("config", DWC_PAD_UNSUPPORTED)
@pytest.mark.fpgadataflow
def test_fpgadataflow_dwc_padding_unsupported(config):
    in_shape, out_shape, inWidth, outWidth, finn_dtype = config
    model = make_single_dwc_modelwrapper(
        in_shape, inWidth, outWidth, finn_dtype, "", out_shape=out_shape
    )
    assert not getCustomOp(model.graph.node[0]).rtl_supported()
