# Copyright (c) 2020-2022, Xilinx
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

import pytest

import copy
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import decompress_string_to_numpy
from finn.util.test import (
    compare_two_chr_funcs,
    debug_chr_funcs,
    get_characteristic_fnc,
    soft_verify_topk,
)


def make_labelselect_modelwrapper(labels, pe, k, idt, impl_style):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, labels])
    outp = helper.make_tensor_value_info("outp", TensorProto.INT64, [1, k])

    labelselect_node = helper.make_node(
        "LabelSelect",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        Labels=labels,
        PE=pe,
        K=k,
        inputDataType=idt.name,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        nodes=[labelselect_node],
        name="graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    odt = DataType.get_smallest_possible(labels - 1)
    model.set_tensor_datatype("outp", odt)

    return model


def prepare_inputs(input_tensor, idt):
    return {"inp": input_tensor}


@pytest.mark.parametrize("idt", [DataType["UINT8"], DataType["UINT16"], DataType["INT16"]])
# labels
@pytest.mark.parametrize("labels", [10, 100])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 10])
# number of top labels to select
@pytest.mark.parametrize("k", [1, 5])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
# impl style
@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_labelselect(idt, labels, fold, k, exec_mode, impl_style):
    np.random.seed(0)
    if fold == -1:
        pe = 1
    else:
        pe = labels // fold
    assert labels % pe == 0

    if k == -1:
        k = labels

    # generate input data
    x = gen_finn_dt_tensor(idt, (1, labels))
    input_dict = prepare_inputs(x, idt)

    model = make_labelselect_modelwrapper(labels, pe, k, idt, impl_style)

    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert soft_verify_topk(x, y, k), "HW layer execution failed"

    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise Exception("Unknown exec_mode")

    y = oxe.execute_onnx(model, input_dict)["outp"]

    assert soft_verify_topk(x, y, k), exec_mode + " failed"


# which port to test
@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("idt", [DataType["UINT8"], DataType["UINT16"], DataType["INT16"]])
# labels
@pytest.mark.parametrize("labels", [10, 100])
# folding
@pytest.mark.parametrize("fold", [-1, 2, 10])
# number of top labels to select
@pytest.mark.parametrize("k", [1, 5])
# impl style
@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_analytical_characterization_labelselect(
    direction, idt, labels, fold, k, impl_style
):
    np.random.seed(0)
    if fold == -1:
        pe = 1
    else:
        pe = labels // fold
    assert labels % pe == 0

    if k == -1:
        k = labels

    model = make_labelselect_modelwrapper(labels, pe, k, idt, impl_style)
    node_details = ("LabelSelect", idt, labels, fold, k, impl_style)
    part = "xc7z020clg400-1"
    target_clk_ns = 4
    allowed_chr_offset_positions = 5

    model_rtl = copy.deepcopy(model)
    node_analytical = get_characteristic_fnc(
        model, (*node_details, "analytical"), part, target_clk_ns, "analytical"
    )
    node_rtlsim = get_characteristic_fnc(
        model_rtl, (*node_details, "rtlsim"), part, target_clk_ns, "rtlsim"
    )

    chr_in = decompress_string_to_numpy(node_analytical.get_nodeattr("io_chrc_in"))
    chr_out = decompress_string_to_numpy(node_analytical.get_nodeattr("io_chrc_out"))

    rtlsim_in = decompress_string_to_numpy(node_rtlsim.get_nodeattr("io_chrc_in"))
    rtlsim_out = decompress_string_to_numpy(node_rtlsim.get_nodeattr("io_chrc_out"))

    debug_chr_funcs(chr_in, chr_out, rtlsim_in, rtlsim_out, direction)

    if direction == "input":
        assert compare_two_chr_funcs(
            chr_in,
            rtlsim_in,
            allowed_chr_offset_positions,
        )
    elif direction == "output":
        assert compare_two_chr_funcs(
            chr_out,
            rtlsim_out,
            allowed_chr_offset_positions,
        )
