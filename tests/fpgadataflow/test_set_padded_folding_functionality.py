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

import numpy as np
import copy
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.util.test import load_test_checkpoint_or_skip

from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
)

from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
import finn.core.onnx_exec as oxe

def make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes,impl_style):
    W = np.random.randint(wdt.min(), wdt.max() + 1, size=(ch, ch))
    W = W.astype(np.float32)

    T = np.random.randint(tdt.min(), tdt.max() + 1, size=(ch, 2 ** adt.bitwidth() - 1))
    T = T.astype(np.float32)

    if impl_style != "stitched_rtl":
        input_name = "inp"
        output_name = "outp"
    else:
        input_name = "inp"
        output_name = "outp"

    tensors = []
    tensors.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, ch]))
    for i in range(1, nnodes):
        inter = helper.make_tensor_value_info("inter_" + str(i), TensorProto.FLOAT, [1, ch])
        tensors.append(inter)
    tensors.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, ch]))

    FCLayer_nodes = []
    for i in range(nnodes):
        pe = 1
        simd = 1
        FCLayer_nodes += [
            helper.make_node(
                "MVAU_hls",
                [tensors[i].name, "weights_" + str(i), "thresh_" + str(i)],
                [tensors[i + 1].name],
                domain="finn.custom_op.fpgadataflow.hls",
                backend="fpgadataflow",
                MW=ch,
                MH=ch,
                SIMD=simd,
                PE=pe,
                inputDataType=adt.name,
                weightDataType=wdt.name,
                outputDataType=adt.name,
                ActVal=0,
                binaryXnorMode=0,
                noActivation=0,
            )
        ]

    graph = helper.make_graph(
        nodes=FCLayer_nodes,
        name="fclayer_graph",
        inputs=[tensors[0]],
        outputs=[tensors[-1]],
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype(input_name, adt)
    model.set_tensor_datatype(output_name, adt)

    for i in range(1, nnodes + 1):
        if tensors[i].name != output_name:
            model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_" + str(i - 1), W)
        model.set_initializer("thresh_" + str(i - 1), T)
        model.set_tensor_datatype("weights_" + str(i - 1), wdt)
        model.set_tensor_datatype("thresh_" + str(i - 1), tdt)

    return model

def prepare_inputs(input_tensor, idt, wdt, inp_name="inp"):
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert bipolar to binary
        return {inp_name: (input_tensor + 1) / 2}
    else:
        return {inp_name: input_tensor}


def generate_thresholding_inputs(adt,wdt,model_naive, model_padded):

    # update thresholding based on the dims

    T_list = []
    T_padded_list = []

    nm_nodes = [getCustomOp(node) for node in model_naive.graph.node if node.op_type in ["MVAU_hls", "MVAU_rtl"]]
    opm_nodes = [getCustomOp(node) for node in model_padded.graph.node if node.op_type in ["MVAU_hls", "MVAU_rtl"]]

    for i in range(len(nm_nodes)):
        nm = nm_nodes[i]
        opm = opm_nodes[i]

        naive_mw = nm.get_nodeattr("MW")
        naive_mh = nm.get_nodeattr("MH")
        padded_mw = opm.get_nodeattr("MW")
        padded_mh = opm.get_nodeattr("MH")

        if adt is None:
            # no activation, produce accumulators
            T = None
            tdt = None
            if wdt == DataType["BIPOLAR"] and adt == DataType["BIPOLAR"]:
                odt = DataType["UINT32"]
            else:
                odt = DataType["INT32"]
        else:
            odt = adt
            (min, max) = calculate_signed_dot_prod_range(adt, wdt, naive_mw)
            n_steps = adt.get_num_possible_values() - 1
            T = np.random.randint(min, max - 1, (padded_mh, n_steps)).astype(np.float32)
            # provide non-decreasing thresholds
            T = np.sort(T, axis=1)
            # generate thresholds for activation
            if wdt == DataType["BIPOLAR"] and adt == DataType["BIPOLAR"]:
                tdt = DataType["UINT32"]
                # bias thresholds to be positive
                T = np.ceil((T + naive_mw) / 2)
                assert (T >= 0).all()
            else:
                tdt = DataType["INT32"]

        T_list.append(T[:naive_mh,:])
        T_padded_list.append(T)
    return T_list, T_padded_list



def apply_new_thresholds(model,T_list):
    for i in range(len(T_list)):
        model.set_initializer(f"thresh_{i}",T_list[i])
    return model


# desired frames per second
@pytest.mark.parametrize("target_fps", [30, 10**5])
# target chip or board
@pytest.mark.parametrize("platform", ["Pynq-Z1", "Ultra96", "U200"])
@pytest.mark.parametrize("impl_style", ["cppsim","stitched_rtlsim","rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_set_padded_folding_functionality(target_fps, platform, impl_style):

    idt = DataType["INT4"]
    wdt = DataType["INT2"]
    adt = DataType["INT16"]
    model = make_multi_fclayer_model(128,idt, wdt, adt, 5,impl_style)

    model = model.transform(GiveUniqueNodeNames())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    clk_ns = 5
    part = "xc7z020clg400-1"
    target_cycles_per_frame = int((10**9 / clk_ns) / target_fps)


    dataflow_model_padded = copy.deepcopy(dataflow_model)
    dataflow_model_naive = dataflow_model.transform(SetFolding(target_cycles_per_frame))
    dataflow_model_padded = dataflow_model_padded.transform(SetFolding(target_cycles_per_frame,padding=6))


    exp_cycles_dict_naive = dataflow_model_naive.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame_naive = max(exp_cycles_dict_naive.values())

    exp_cycles_dict_padded = dataflow_model_padded.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame_padded = max(exp_cycles_dict_padded.values())

    min_cycles = dict()
    min_cycles["Pynq-Z1"] = 128
    min_cycles["Ultra96"] = 64
    min_cycles["U200"] = 1

    # evaluate that cycles are still within the requirements
    assert achieved_cycles_per_frame_naive <= max(
        min_cycles[platform], target_cycles_per_frame
    ), "Naive Folding target not met"
    assert achieved_cycles_per_frame_padded <= max(
        min_cycles[platform], target_cycles_per_frame
    ), "Padded Folding target not met"

    # evaluate that the functionality is still identical

    # preparing two identical, sensible thresholding inputs with padding applied to the second
    T_naive_list, T_padded_list = generate_thresholding_inputs(idt,wdt,dataflow_model_naive, dataflow_model_padded)
    dataflow_model_naive = apply_new_thresholds(dataflow_model_naive,T_naive_list)
    dataflow_model_padded = apply_new_thresholds(dataflow_model_padded,T_padded_list)

    input_mw_naive = getCustomOp(dataflow_model_naive.graph.node[0]).get_nodeattr("MW")
    output_mh_naive = getCustomOp(dataflow_model_naive.graph.node[-1]).get_nodeattr("MH")

    input_mw_padded = getCustomOp(dataflow_model_padded.graph.node[0]).get_nodeattr("MW")
    output_mh_padded = getCustomOp(dataflow_model_padded.graph.node[-1]).get_nodeattr("MH")
    x = np.zeros((1,input_mw_padded),dtype=np.float32)
    y = np.zeros((1,output_mh_padded),dtype=np.float32)


    if impl_style == "cppsim":

        input_name = "inp"
        output_name = "outp"

       # dataflow_model_padded.set_initializer(input_name, x)
      #  dataflow_model_padded.set_initializer(output_name, y)

        dataflow_model_naive = dataflow_model_naive.transform(SpecializeLayers())
        dataflow_model_naive = dataflow_model_naive.transform(GiveUniqueNodeNames())
        dataflow_model_naive = dataflow_model_naive.transform(AnnotateCycles())
        dataflow_model_naive = dataflow_model_naive.transform(SetExecMode("cppsim"))
        dataflow_model_naive = dataflow_model_naive.transform(PrepareCppSim())
        dataflow_model_naive = dataflow_model_naive.transform(CompileCppSim())


        dataflow_model_padded = dataflow_model_padded.transform(SpecializeLayers())
        dataflow_model_padded = dataflow_model_padded.transform(GiveUniqueNodeNames())
        dataflow_model_padded = dataflow_model_padded.transform(AnnotateCycles())
        dataflow_model_padded = dataflow_model_padded.transform(SetExecMode("cppsim"))
        dataflow_model_padded = dataflow_model_padded.transform(PrepareCppSim())
        dataflow_model_padded = dataflow_model_padded.transform(CompileCppSim())

    elif impl_style == "rtlsim":

        input_name = "inp"
        output_name = "outp"

       # dataflow_model_padded.set_initializer(input_name, x)
       # dataflow_model_padded.set_initializer(output_name, y)

        dataflow_model_naive = dataflow_model_naive.transform(SpecializeLayers(part))
        dataflow_model_naive = dataflow_model_naive.transform(GiveUniqueNodeNames())
        dataflow_model_naive = dataflow_model_naive.transform(AnnotateCycles())
        dataflow_model_naive = dataflow_model_naive.transform(SetExecMode("rtlsim"))
        dataflow_model_naive = dataflow_model_naive.transform(PrepareIP(part, clk_ns))
        dataflow_model_naive = dataflow_model_naive.transform(HLSSynthIP())
        dataflow_model_naive = dataflow_model_naive.transform(PrepareRTLSim())

        dataflow_model_padded = dataflow_model_padded.transform(SpecializeLayers(part))
        dataflow_model_padded = dataflow_model_padded.transform(GiveUniqueNodeNames())
        dataflow_model_padded = dataflow_model_padded.transform(AnnotateCycles())
        dataflow_model_padded = dataflow_model_padded.transform(SetExecMode("rtlsim"))
        dataflow_model_padded = dataflow_model_padded.transform(PrepareIP(part, clk_ns))
        dataflow_model_padded = dataflow_model_padded.transform(HLSSynthIP())
        dataflow_model_padded = dataflow_model_padded.transform(PrepareRTLSim())


    elif impl_style == "stitched_rtlsim":

        input_name = "inp"
        output_name = "outp"


        #dataflow_model_padded.set_initializer(input_name, x)
        #dataflow_model_padded.set_initializer(output_name, y)

       # if (input_mw_padded != input_mw_naive):
        if len(dataflow_model_padded.graph.input) != 0:
            dataflow_model_padded.graph.input.remove(dataflow_model_padded.graph.input[0])
        input_x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, input_mw_padded])
        dataflow_model_padded.graph.input.append(input_x)
       # if (input_mw_padded != input_mw_naive):
       #     dataflow_model_padded.set_initializer(input_name, x)

        #if (output_mh_padded != output_mh_naive):
        if len(dataflow_model_padded.graph.output) != 0:
            dataflow_model_padded.graph.output.remove(dataflow_model_padded.graph.output[0])
        output_y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, output_mh_padded])
        dataflow_model_padded.graph.output.append(output_y)
       # if (output_mh_padded != output_mh_naive):
       #     dataflow_model_padded.set_initializer(output_name, y)
        
    

        dataflow_model_naive = dataflow_model_naive.transform(SpecializeLayers(part))
        dataflow_model_naive = dataflow_model_naive.transform(GiveUniqueNodeNames())
        dataflow_model_naive = dataflow_model_naive.transform(SetExecMode("rtlsim"))
        dataflow_model_naive = dataflow_model_naive.transform(PrepareIP(part, clk_ns))
        dataflow_model_naive = dataflow_model_naive.transform(HLSSynthIP())
        dataflow_model_naive = dataflow_model_naive.transform(PrepareRTLSim())
        dataflow_model_naive = dataflow_model_naive.transform(InsertAndSetFIFODepths(part, clk_ns))
        dataflow_model_naive = dataflow_model_naive.transform(PrepareIP(part, clk_ns))
        dataflow_model_naive = dataflow_model_naive.transform(HLSSynthIP())
        dataflow_model_naive = dataflow_model_naive.transform(CreateStitchedIP(part, clk_ns))
        dataflow_model_naive.set_metadata_prop("rtlsim_so", "")
        dataflow_model_naive.set_metadata_prop("exec_mode", "rtlsim")


        dataflow_model_padded = dataflow_model_padded.transform(SpecializeLayers(part))
        dataflow_model_padded = dataflow_model_padded.transform(GiveUniqueNodeNames())
        dataflow_model_padded = dataflow_model_padded.transform(SetExecMode("rtlsim"))
        dataflow_model_padded = dataflow_model_padded.transform(PrepareIP(part, clk_ns))
        dataflow_model_padded = dataflow_model_padded.transform(HLSSynthIP())
        dataflow_model_padded = dataflow_model_padded.transform(PrepareRTLSim())
        dataflow_model_padded = dataflow_model_padded.transform(InsertAndSetFIFODepths(part, clk_ns))
        dataflow_model_padded = dataflow_model_padded.transform(PrepareIP(part, clk_ns))
        dataflow_model_padded = dataflow_model_padded.transform(HLSSynthIP())
        dataflow_model_padded = dataflow_model_padded.transform(CreateStitchedIP(part, clk_ns))

        dataflow_model_padded.set_metadata_prop("rtlsim_so", "")
        dataflow_model_padded.set_metadata_prop("exec_mode", "rtlsim")

        
        # we have to adjust the global_in and global_out if padding happened



   # dataflow_model_padded.set_initializer(input_name, x)
  #  dataflow_model_padded.set_initializer(output_name, y)


    # preparing two identical, sensible input vectors with padding applied to the second

    x_input_padded = gen_finn_dt_tensor(idt, (1, input_mw_padded))
    input_dict_padded = prepare_inputs(x_input_padded, idt, wdt,inp_name=input_name)
    
    input_mw_naive = getCustomOp(dataflow_model_naive.graph.node[0]).get_nodeattr("MW")
    x_input_naive = gen_finn_dt_tensor(idt, (1, input_mw_naive))
    x_input_naive[:,:input_mw_naive] = x_input_padded[:,:input_mw_naive]
    input_dict_naive = prepare_inputs(x_input_naive, idt, wdt,inp_name=input_name)



    y_naive = oxe.execute_onnx(dataflow_model_naive, input_dict_naive)
    y_naive = y_naive[dataflow_model_naive.graph.output[0].name]
    y_naive = y_naive.reshape(dataflow_model_naive.get_tensor_shape(output_name))

    y_padded = oxe.execute_onnx(dataflow_model_padded, input_dict_padded)
    y_padded = y_padded[dataflow_model_padded.graph.output[0].name]
    y_padded = y_padded.reshape(dataflow_model_padded.get_tensor_shape(output_name))


    output_mh_naive = getCustomOp(dataflow_model_naive.graph.node[-1]).get_nodeattr("MH")

    y_padded = y_padded[:,:output_mh_naive]

    assert np.array_equal(y_naive, y_padded)
