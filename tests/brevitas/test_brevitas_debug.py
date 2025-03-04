# Copyright (c) 2020, Xilinx
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

import brevitas.onnx as bo
import numpy as np
import onnx
import onnx.numpy_helper as nph
import os
import torch
from brevitas.export import export_qonnx
from brevitas.quant_tensor import _unpack_quant_tensor
from pkgutil import get_data
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.core.onnx_exec as oxe
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.basic import make_build_dir
from finn.util.test import get_test_model_trained


@pytest.mark.brevitas_export
@pytest.mark.parametrize("QONNX_FINN_conversion", [False, True])
def test_brevitas_debug(QONNX_FINN_conversion):
    build_dir = make_build_dir("test_brevitas_debug")
    finn_onnx = os.path.join(build_dir, "test_brevitas_debug.onnx")
    fc = get_test_model_trained("TFC", 2, 2)
    ishape = (1, 1, 28, 28)
    dbg_hook = bo.enable_debug(fc, proxy_level=True)
    export_qonnx(fc, torch.randn(ishape), finn_onnx)
    # DebugMarkers have the brevitas.onnx domain, so that needs adjusting
    model = ModelWrapper(finn_onnx)
    dbg_nodes = model.get_nodes_by_op_type("DebugMarker")
    for dbg_node in dbg_nodes:
        dbg_node.domain = "qonnx.custom_op.general"
    model.save(finn_onnx)
    qonnx_cleanup(finn_onnx, out_file=finn_onnx)
    if QONNX_FINN_conversion:
        model = ModelWrapper(finn_onnx)
        model = model.transform(ConvertQONNXtoFINN())
        model.save(finn_onnx)
    model = ModelWrapper(finn_onnx)
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1
    # load one of the test vectors
    raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
    input_tensor = onnx.load_tensor_from_string(raw_i)
    # run using FINN-based execution
    input_dict = {model.graph.input[0].name: nph.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model, input_dict, return_full_exec_context=True)
    produced = output_dict[model.graph.output[0].name]
    # run using PyTorch/Brevitas
    input_tensor = torch.from_numpy(nph.to_array(input_tensor)).float()
    assert input_tensor.shape == (1, 1, 28, 28)
    # do forward pass in PyTorch/Brevitas
    expected = fc.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()
    # check all tensors at debug markers
    names_brevitas = set(dbg_hook.values.keys())
    names_finn = set(output_dict.keys())
    names_common = names_brevitas.intersection(names_finn)
    # The different exports return debug markers in different numbers and places
    print(len(names_common))
    if not QONNX_FINN_conversion:
        assert len(names_common) == 12
    else:
        assert len(names_common) == 8
    for dbg_name in names_common:
        tensor_pytorch = _unpack_quant_tensor(dbg_hook.values[dbg_name]).detach().numpy()
        tensor_finn = output_dict[dbg_name]
        assert np.isclose(tensor_finn, tensor_pytorch, atol=1e-5).all()
