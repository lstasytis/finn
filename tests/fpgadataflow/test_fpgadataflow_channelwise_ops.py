# Copyright (c) 2020-2022, Xilinx, Inc.
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
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import decompress_string_to_numpy
from finn.util.test import compare_two_chr_funcs, get_characteristic_fnc


def make_modelwrapper(C, pe, idt, odt, pdt, func, vecs):
    NumChannels = C.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, vecs + [NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, vecs + [NumChannels])

    node_inp_list = ["inp", "const"]

    node = helper.make_node(
        "ChannelwiseOp",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        Func=func,
        PE=pe,
        inputDataType=idt.name,
        outputDataType=odt.name,
        paramDataType=pdt.name,
        numInputVectors=vecs,
        preferred_impl_style="hls",
    )
    graph = helper.make_graph(nodes=[node], name="graph", inputs=[inp], outputs=[outp])

    model = qonnx_make_model(graph, producer_name="model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("const", idt)
    model.set_initializer("const", C)
    return model


# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["INT8"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT4"]])
# param datatype
@pytest.mark.parametrize("pdt", [DataType["INT4"]])
# folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2])
# number of input features
@pytest.mark.parametrize("ich", [16])
# vecs
@pytest.mark.parametrize("vecs", [[1], [1, 7, 7]])
# function
@pytest.mark.parametrize("func", ["add", "mul"])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_channelwise_ops(idt, act, pdt, nf, ich, func, vecs, exec_mode):
    if nf == -1:
        nf = ich
    pe = ich // nf
    assert ich % pe == 0

    # generate input and param data
    x = gen_finn_dt_tensor(idt, tuple(vecs + [ich]))
    C = gen_finn_dt_tensor(pdt, (ich))

    odt = act

    # create model
    model = make_modelwrapper(C, pe, idt, odt, pdt, func, vecs)

    # package input data as dictionary
    input_dict = {"inp": x}

    oshape = model.get_tensor_shape("outp")

    C_reshaped = np.broadcast_to(C.flatten(), x.shape)
    if func == "add":
        y = x + C_reshaped
    elif func == "mul":
        y = x * C_reshaped

    y_expected = y.reshape(oshape)

    # verify hw abstraction layer
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), "HW layer execution failed"

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

    # execute model
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]

    y_produced = y_produced.reshape(y_expected.shape)

    assert (y_produced == y_expected).all(), exec_mode + " failed"

    if exec_mode == "rtlsim":
        hls_synt_res_est = model.analysis(hls_synth_res_estimation)
        assert "ChannelwiseOp_hls_0" in hls_synt_res_est

        node = model.get_nodes_by_op_type("ChannelwiseOp_hls")[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0


# which port to test
@pytest.mark.parametrize("direction", ["input", "output"])
# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["INT8"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT4"]])
# param datatype
@pytest.mark.parametrize("pdt", [DataType["INT4"]])
# folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1, 2])
# number of input features
@pytest.mark.parametrize("ich", [16])
# vecs
@pytest.mark.parametrize("vecs", [[1], [1, 7, 7]])
# function
@pytest.mark.parametrize("func", ["add", "mul"])
# execution mode
@pytest.mark.parametrize("exec_mode", ["rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_fpgadataflow_analytical_characterization_channelwise_ops(
    direction, idt, act, pdt, nf, ich, func, vecs, exec_mode
):
    if nf == -1:
        nf = ich
    pe = ich // nf
    assert ich % pe == 0

    # generate param data
    C = gen_finn_dt_tensor(pdt, (ich))

    odt = act

    # create model
    model = make_modelwrapper(C, pe, idt, odt, pdt, func, vecs)
    node_details = ("ChannelWiseOp", C, pe, idt, odt, pdt, func, "hls")
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

    # DEBUGGING ======================================================
    if direction == "input":
        np.set_printoptions(threshold=np.inf)
        print("chr IN")
        print(chr_in[:100])

        print("rtlsim IN")
        print(rtlsim_in[:100])

    elif direction == "output":
        np.set_printoptions(threshold=np.inf)
        print("chr OUT")
        print(chr_out[:100])

        print("rtlsim OUT")
        print(rtlsim_out[:100])
    # DEBUGGING ======================================================

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
