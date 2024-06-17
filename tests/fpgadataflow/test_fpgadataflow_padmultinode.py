# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
from finn.core.throughput_test import throughput_test_rtlsim
from finn.util.pyverilator import verilator_fifosim
import pytest
import copy
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)

from finn.transformation.fpgadataflow.vitis_build import CreateVitisXO

from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)


from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.util.test import load_test_checkpoint_or_skip

import numpy as np
import qonnx.custom_op.general.xnorpopcount as xp
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)

from qonnx.transformation.infer_shapes import InferShapes

import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers



def make_multi_fclayer_model(mw,mh, wdt, adt, tdt, nnodes):

    tensors = []
    tensors.append(helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw[0]]))
    for i in range(1, nnodes):
        inter = helper.make_tensor_value_info("inter_" + str(i), TensorProto.FLOAT, [1, mw[i]])
        tensors.append(inter)
    tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh[-1]]))


    Wl = []
    Tl = []

    FCLayer_nodes = []
    for i in range(nnodes):


        W = np.random.randint(wdt.min(), wdt.max() + 1, size=(mw[i], mh[i]))
        W = W.astype(np.float32)

        T = np.random.randint(tdt.min(), tdt.max() + 1, size=(mh[i], 2 ** adt.bitwidth() - 1))
        T = T.astype(np.float32)


        Wl.append(W)
        Tl.append(T)

        pe = 1
        simd = 1
        FCLayer_nodes += [
            helper.make_node(
                "MVAU_hls",
                [tensors[i].name, "weights_" + str(i), "thresh_" + str(i)],
                [tensors[i + 1].name],
                domain="finn.custom_op.fpgadataflow.hls",
                backend="fpgadataflow",
                MW=mw[i],
                MH=mh[i],
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

    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)



    for i in range(1, nnodes + 1):
        if tensors[i].name != "outp":
            model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_" + str(i - 1), Wl[i-1])
        model.set_initializer("thresh_" + str(i - 1), Tl[i-1])
        model.set_tensor_datatype("weights_" + str(i - 1), wdt)
        model.set_tensor_datatype("thresh_" + str(i - 1), tdt)

  #  model = model.transform(GiveUniqueNodeNames())
  #  parent_model = model.transform(CreateDataflowPartition())
  #  sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
  #  sdp_node = getCustomOp(sdp_node)
  #  dataflow_model_filename = sdp_node.get_nodeattr("model")
   # dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    return model



def prepare_inputs(input_tensor, idt, wdt, inp_name="inp"):
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert bipolar to binary
        return {inp_name: (input_tensor + 1) / 2}
    else:
        return {inp_name: input_tensor}


def update_mvau_nodes(model,mw,mh,simd,pe,W,T, impl_style):


   # tensors = []
   # tensors.append(helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw[0]]))
   # for i in range(1, len(mw)):
   #     inter = helper.make_tensor_value_info("inter_" + str(i), TensorProto.FLOAT, [1, mw[i]])
   #     tensors.append(inter)
   # tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh[-1]]))


    for i in range(len(mw)):

        
        # fetch the node
        node_inst = getCustomOp(model.graph.node[i])

    

        #update mh and mw in case of padding
        node_inst.set_nodeattr("MW", mw[i])
        node_inst.set_nodeattr("MH", mh[i])

        #update simd and pe in case of padding
        node_inst.set_nodeattr("SIMD", simd[i])
        node_inst.set_nodeattr("PE", pe[i])


        # initialize new padded or cropped tensors for the node inputs and outputs

        model.set_initializer(model.graph.node[i].input[1], W[i])
        model.set_initializer(model.graph.node[i].input[2], T[i])

        x = np.zeros((1,mw[i]),dtype=np.float32)
        y = np.zeros((1,mh[i]),dtype=np.float32)

        if i != 0:
            model.set_initializer(model.graph.node[i].input[0], x)
        #else:
        #    if impl_style == "stitched_rtlsim":
        #        model.set_initializer("global_in", x)

        if i != len(mw)-1:
            model.set_initializer(model.graph.node[i].output[0], y)
        #else:
        #    if impl_style == "stitched_rtlsim":
        #        model.set_initializer("global_out", y)


    # insert DWCs, mandatory when padding or cropping
    model = model.transform(InsertDWC())

    # at this point, if padding_output != 0,
    # we could insert a DWC at the end as well
    # to avoid having to crop in software

    part = "xc7z020clg400-1"
    clk_ns = 1.66

    input_name = "inp"
    output_name = "outp"



    if impl_style == "cppsim":
        model = model.transform(SpecializeLayers())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        

    elif impl_style == "rtlsim":
        model = model.transform(SpecializeLayers(part))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(AnnotateCycles())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(part, clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

        model.set_metadata_prop("rtlsim_so", "")
        model.set_metadata_prop("exec_mode", "rtlsim")




    elif impl_style == "stitched_rtlsim":
        model = model.transform(SpecializeLayers(part))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SetExecMode("rtlsim"))

        model = model.transform(PrepareIP(part, clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

        model = model.transform(InsertAndSetFIFODepths(part, clk_ns))
        model = model.transform(PrepareIP(part, clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(CreateStitchedIP(part, clk_ns))
       # model = model.transform(PrepareRTLSim())

        model.set_metadata_prop("rtlsim_so", "")
        model.set_metadata_prop("exec_mode", "rtlsim")

    elif impl_style == "xo":
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(CreateDataflowPartition())
        model.set_metadata_prop("rtlsim_so", "")
        model.set_metadata_prop("exec_mode", "rtlsim")

        kernel_model = model
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        for sdp_node in sdp_nodes:
            prefix = sdp_node.name + "_"
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_model = kernel_model.transform(InsertFIFO())
            kernel_model = kernel_model.transform(SpecializeLayers())
            kernel_model = kernel_model.transform(RemoveUnusedTensors())
            kernel_model = kernel_model.transform(GiveUniqueNodeNames())
            kernel_model.save(dataflow_model_filename)
            kernel_model = kernel_model.transform(PrepareIP(part, clk_ns))
            kernel_model = kernel_model.transform(HLSSynthIP())
            kernel_model = kernel_model.transform(
                CreateStitchedIP(part, clk_ns, sdp_node.onnx_node.name, True)
            )
            kernel_model.set_metadata_prop("platform", "alveo")
            kernel_model = kernel_model.transform(CreateVitisXO(sdp_node.onnx_node.name),cleanup=True)
            
            kernel_model.save(dataflow_model_filename)
        model = kernel_model
        model.set_metadata_prop("rtlsim_so", "")
        model.set_metadata_prop("exec_mode", "rtlsim")

    x = np.zeros((1,mw[0]),dtype=np.float32)
    y = np.zeros((1,mh[-1]),dtype=np.float32)

    if impl_style in ["stitched_rtlsim","xo"]:
        model.set_initializer("global_in", x)
        model.set_initializer("global_out", y)
   # else:
    #    model.set_initializer(model.graph.node[0].input[0], x)
    #    model.set_initializer(model.graph.node[-1].output[0], y)

    #model = model.transform(InferShapes())
   # model = model.transform(InferDataTypes())


    input_mw_naive = getCustomOp(model.graph.node[0]).get_normal_input_shape()
    output_mh_naive = getCustomOp(model.graph.node[-1]).get_normal_output_shape()

    if len(model.graph.input) != 0:
        model.graph.input.remove(model.graph.input[0])
    input_x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [*input_mw_naive])
    model.graph.input.append(input_x)
    
    
    # if (input_mw_padded != input_mw_naive):in
    #     dataflow_model_padded.set_initializer(input_name, x)

    #if (output_mh_padded != output_mh_naive):
    if len(model.graph.output) != 0:
        model.graph.output.remove(model.graph.output[0])
    output_y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [*output_mh_naive])
    model.graph.output.append(output_y)

   # model.get_tensor_shape("inp", fix_missing_init_shape=True)
  #  model.get_tensor_shape("outp", fix_missing_init_shape=True)

    model.set_tensor_shape("inp",input_mw_naive)
    model.set_tensor_shape("outp",output_mh_naive)

    return model


# activation: None or DataType
@pytest.mark.parametrize("act", [DataType["INT2"]])
# weight datatype
@pytest.mark.parametrize("wdt", [DataType["INT2"]])
# input datatype
@pytest.mark.parametrize("idt", [DataType["INT2"]])
# neuron folding, -1 is maximum possible
@pytest.mark.parametrize("nf", [-1])
# synapse folding, -1 is maximum possible
@pytest.mark.parametrize("sf", [-1])

@pytest.mark.parametrize("dims",[


    (
    # DWC when in width > out width
    [1,6,0,0,1,2], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,1,0,0,3,1]),  

    (
    # DWC when in width > out width
    [2,24,0,0,1,6], # mw,mh,mw_padding, mh_padding, simd, pe
    [24,2,1,0,5,1]),  
    ( 
    # pad first node output and second node input by 1
    # DWC where in width < out width
    [4,5,0,1,1,3], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,1,0,7,1]),
    # each list is an individual MVAU node




    ( 
    # pad first node output and second node input by 1
    # DWC where in width < out width
    [4,5,0,1,1,3], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,1,0,7,4],
    [4,2,0,0,2,1]),    
    ( 
    # pad first node output by 2 and second node input by 1
    # DWC where in width < out width
    [4,5,0,2,1,3], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,1,0,7,1]),  
    (
    # pad first node output by 1 and second node input by 2
    # DWC where in width < out width
    [4,5,0,1,1,3], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,2,0,8,1]),  
    ( 
    # pad second node input by 1
    # DWC where in width < out width
    [4,6,0,0,1,3], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,1,0,7,1]),  
    (
    # pad first node input and second node output by 1
    # DWC when in width < out width
    [2,24,1,0,1,4], # mw,mh,mw_padding, mh_padding, simd, pe
    [24,2,0,1,6,1]),  

    ( 
    # DWC when in width < out width
    [4,6,0,0,1,6], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,3,1]),  
    ( 
    [4,6,0,0,1,2], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,2,0,4,1]),      
    ( 
    [4,6,0,0,1,3], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,1,1]),  
    (
    [4,24,0,0,1,2], # mw,mh,mw_padding, mh_padding, simd, pe
    [24,4,0,0,6,1]),  
    ( 
    #
    [4,6,0,0,1,1], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,3,1]),      
    (
    #
    [1,24,0,0,1,4], # mw,mh,mw_padding, mh_padding, simd, pe
    [24,1,0,0,6,1]),  
    ( 
    #
    [4,6,0,0,1,1], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,1,1]),  
    ( 
    #
    [4,6,1,0,1,1], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,1,1]),  
    ( 
    #
    [4,6,0,1,1,1], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,1,1]),  
    ( 
    # pad 2nd node input
    [4,6,0,0,1,1], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,1,1,1]),  
    ( 
    # pad first node output
    [4,6,0,1,1,1], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,1,2]),  
    ( 
    # pad second node input
    [4,6,0,0,2,1], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,1,0,1,1]),  
    ( 
    # pad first node output by 2
    # DWC where in width > out width
    [4,6,0,2,1,4], # mw,mh,mw_padding, mh_padding, simd, pe
    [6,4,0,0,2,1]),  
    ])
@pytest.mark.parametrize("impl_style", ["rtlsim","stitched_rtlsim","cppsim","xo"])
@pytest.mark.parametrize("perf_test", [False])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_fpgadataflow_mvau_multiple_nodes_padded_folded_dwc_inserted(idt, wdt, act, nf, sf, dims, impl_style,perf_test):

    nnodes = len(dims)

    input_mw = dims[0][0]
    input_mw_padding = dims[0][2]
    x = gen_finn_dt_tensor(idt, (1, input_mw))


    # generate padded input data
    x_padded = gen_finn_dt_tensor(idt, (1, input_mw+input_mw_padding))
    x_padded[...] = 0
    x_padded[:,:input_mw] = x[:,:input_mw]

    mw_no_padding = [x[0] for x in dims]
    mh_no_padding = [x[1] for x in dims]


    # generate all the necessary node parameter lists given the input
    W_padded_list = []
    W_list = []
    pe_list = []
    simd_list = []
    pe_padded_list = []
    simd_padded_list = []
    mw_padded = []
    mh_padded = []
    for i in range(nnodes):

        mw = dims[i][0]
        mh = dims[i][1]

        mw_padding = dims[i][2]
        mh_padding = dims[i][3]

        simd = dims[i][4]
        pe = dims[i][5]

        mw_padded.append(mw+mw_padding)
        mh_padded.append(mh+mh_padding)



        W = gen_finn_dt_tensor(wdt, (mw, mh))


        W_padded = gen_finn_dt_tensor(wdt, (mw+mw_padding, mh+mh_padding))
        W_padded[...] = 0


        simd_padded = simd
        pe_padded = pe

        if sf == -1:
            simd_naive = mw
        else:
            simd_naive = 1 
        if nf == -1:
            pe_naive = mh
        else:
            pe_naive = 1


        W_padded[:mw,:mh] = W[:mw,:mh]
        W_padded_list.append(W_padded)
        W_list.append(W)
        pe_list.append(pe_naive)
        simd_list.append(simd_naive)
        pe_padded_list.append(pe_padded)
        simd_padded_list.append(simd_padded)

    T_list = []
    T_padded_list = []
    for i in range(nnodes):
        if act is None:
            # no activation, produce accumulators
            T = None
            tdt = None
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                odt = DataType["UINT32"]
            else:
                odt = DataType["INT32"]
        else:
            odt = act
            (min, max) = calculate_signed_dot_prod_range(idt, wdt, dims[i][0])
            n_steps = act.get_num_possible_values() - 1
            T = np.random.randint(min, max - 1, (dims[i][1]+dims[i][3], n_steps)).astype(np.float32)
            # provide non-decreasing thresholds
            T = np.sort(T, axis=1)
            # generate thresholds for activation
            if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
                tdt = DataType["UINT32"]
                # bias thresholds to be positive
                T = np.ceil((T + dims[i][0]) / 2)
                assert (T >= 0).all()
            else:
                tdt = DataType["INT32"]

        T_list.append(T[:dims[i][1],:])
        T_padded_list.append(T)

    # create the ground truth model
    model = make_multi_fclayer_model(mw_no_padding,mh_no_padding, wdt, idt, tdt, nnodes)

    model_true = copy.deepcopy(model)
    model_padded = copy.deepcopy(model)

    # update ground truth with the generated weights, activations, impl style and pe & simd
    model_true = update_mvau_nodes(model_true,mw_no_padding,mh_no_padding,
                                   simd_list,pe_list,W_list,T_list,impl_style)


    input_dict = prepare_inputs(x, idt, wdt,inp_name=model_true.graph.input[0].name)
    y_true = oxe.execute_onnx(model_true, input_dict)
    y_true = y_true[model_true.graph.output[0].name]
    y_true = y_true.reshape(model_true.get_tensor_shape("outp"))

   # update padded model with the generated weights, activations, impl style and pe & simd
    model_padded = update_mvau_nodes(model_padded,mw_padded,mh_padded,
                                     simd_padded_list,pe_padded_list,W_padded_list,T_padded_list,impl_style)

    input_dict_padded = prepare_inputs(x_padded, idt, wdt,inp_name=model_padded.graph.input[0].name)
    y_padded = oxe.execute_onnx(model_padded, input_dict_padded)
    y_padded = y_padded[model_padded.graph.output[0].name]
    y_padded = y_padded.reshape(model_padded.get_tensor_shape("outp"))

    # crop output's padding if it exists (this may also be done with a DWC in hardware)
    if dims[-1][3] > 0:
        y_padded = y_padded[:,:-dims[-1][3]]


    assert np.array_equal(y_true, y_padded)


    if perf_test:
        rtlsim_bs = 50
        res_naive = throughput_test_rtlsim(model_true, rtlsim_bs)
        res_padded = throughput_test_rtlsim(model_padded, rtlsim_bs)
        print("naive perf:")
        print(res_naive)
        print("padded perf:")
        print(res_padded)
        assert True == False
