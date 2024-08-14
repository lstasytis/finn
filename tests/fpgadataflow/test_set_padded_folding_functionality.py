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

import copy
import numpy as np
import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.im2col import compute_conv_output_dim
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.util.basic import (
    calculate_signed_dot_prod_range,
    gen_finn_dt_tensor,
    qonnx_make_model,
)
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.fpgadataflow import is_fpgadataflow_node
from finn.util.test import load_test_checkpoint_or_skip


def make_single_im2col_modelwrapper(k, ifm_ch, ifm_dim, ofm_dim, stride, dilation, idt, dw):
    k_h, k_w = k
    ifm_dim_h, ifm_dim_w = ifm_dim
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    ofm_dim_h, ofm_dim_w = ofm_dim

    odt = idt
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm_dim_h, ifm_dim_w, ifm_ch])
    outp = helper.make_tensor_value_info(
        "outp", TensorProto.FLOAT, [1, ofm_dim_h, ofm_dim_w, k_h * k_w * ifm_ch]
    )

    im2col_node = helper.make_node(
        "Im2Col",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.general",
        stride=[stride_h, stride_w],
        kernel_size=[k_h, k_w],
        input_shape=str((1, ifm_dim_h, ifm_dim_w, ifm_ch)),
        dilations=[dilation_h, dilation_w],
        pad_amount=[0, 0, 0, 0],
        pad_value=0,
        depthwise=dw,
    )
    graph = helper.make_graph(
        nodes=[im2col_node], name="im2col_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="im2col-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    return model


def generate_random_threshold_values(input_data_type, num_input_channels, num_steps):
    return np.random.randint(
        input_data_type.min(),
        input_data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)


def sort_thresholds_increasing(thresholds):
    return np.sort(thresholds, axis=1)


# n = batch, c = channel, h = height, w = width of feature map
# Standard = NCHW; FINN = NHWC
# Convert from NHWC(FINN) to NCHW(Standard)
def layout_FINN2NCHW(data):
    return np.transpose(data, (0, 3, 1, 2))


# Convert from NCHW(Standard) to NHWC(FINN)
def layout_NCHW2FINN(data):
    return np.transpose(data, (0, 2, 3, 1))


def make_single_thresholding_modelwrapper(impl_style, T, idt, odt, actval, n_inp_vecs):
    NumChannels = T.shape[0]

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, n_inp_vecs + [NumChannels])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, n_inp_vecs + [NumChannels])

    node_inp_list = ["inp", "thresh"]

    Thresholding_node = helper.make_node(
        "Thresholding",
        node_inp_list,
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        NumChannels=NumChannels,
        numSteps=T.shape[1],
        inputDataType=idt.name,
        weightDataType=idt.name,  # will be set by MinimizeAccumulatorWidth
        outputDataType=odt.name,
        ActVal=actval,
        numInputVectors=n_inp_vecs,
        preferred_impl_style=impl_style,
    )
    graph = helper.make_graph(
        nodes=[Thresholding_node],
        name="thresholding_graph",
        inputs=[inp],
        outputs=[outp],
    )

    model = qonnx_make_model(graph, producer_name="thresholding-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", idt)
    model.set_tensor_datatype("outp", odt)

    model.set_tensor_datatype("thresh", idt)
    model.set_initializer("thresh", T)
    return model


def make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes, impl_style):
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


def generate_thresholding_inputs(adt, wdt, model_naive, model_padded):
    # update thresholding based on the dims

    T_list = []
    T_padded_list = []

    nm_nodes = [
        getCustomOp(node)
        for node in model_naive.graph.node
        if node.op_type in ["MVAU_hls", "MVAU_rtl"]
    ]
    opm_nodes = [
        getCustomOp(node)
        for node in model_padded.graph.node
        if node.op_type in ["MVAU_hls", "MVAU_rtl"]
    ]

    for i in range(len(nm_nodes)):
        nm = nm_nodes[i]
        opm = opm_nodes[i]

        naive_mw = nm.get_nodeattr("MW")
        naive_mh = nm.get_nodeattr("MH")
        padded_mh = opm.get_nodeattr("MH")

        if adt is None:
            # no activation, produce accumulators
            T = None
            tdt = None
        else:
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

        T_list.append(T[:naive_mh, :])
        T_padded_list.append(T)
    return T_list, T_padded_list


def apply_new_thresholds(model, T_list):
    for i in range(len(T_list)):
        model.set_initializer(f"thresh_{i}", T_list[i])
    return model


def update_model(model, part):
    model = model.transform(InsertDWC())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(SpecializeLayers(part))
    model = model.transform(AnnotateCycles())
    return model


@pytest.mark.parametrize("target_fps", [300])
# target chip or board
@pytest.mark.parametrize("platform", ["Pynq-Z1"])


#@pytest.mark.parametrize(
#    "model_type", ["cnv","convinputgenerator-only", "threshold-only","mvau-only"]
#)
#@pytest.mark.parametrize("exec_mode", ["rtlsim","cppsim", "stitched_rtlsim"])

@pytest.mark.parametrize(
    "model_type", ["mvau-only"]
)
@pytest.mark.parametrize("exec_mode", ["cppsim"])
@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_set_padded_folding_functionality(target_fps, model_type, exec_mode, platform, impl_style):
    idt = DataType["INT4"]
    wdt = DataType["INT2"]
    adt = DataType["INT16"]
    build_dir = os.environ["FINN_BUILD_DIR"]

    # TODO: this has to be changed to whatever is the platform's part.
    part = "xc7z020clg400-1"

    if model_type == "convinputgenerator-only":
        conv_config = (1, 2, 0)
        depthwise = False  # True
        use_rtl_swg = True  # True
        kernel_size, stride, pad = conv_config
        np.random.seed(0)
        idt = DataType["UINT4"]

        in_feature_dim = 7
        in_chn = 16

        if use_rtl_swg and exec_mode == "cppsim":
            pytest.skip("Skip cppsim if SWG in rtl")

        if depthwise is True:
            group = out_chn = in_chn
            conv_param_shape = [out_chn, 1, kernel_size, kernel_size]
        else:
            group = 1
            out_chn = 20
            conv_param_shape = [out_chn, in_chn, kernel_size, kernel_size]

        total_pad = 2 * pad
        out_feature_dim = compute_conv_output_dim(in_feature_dim, kernel_size, stride, total_pad)

        input_shape = [1, in_chn, in_feature_dim, in_feature_dim]
        output_shape = [1, out_chn, out_feature_dim, out_feature_dim]

        conv_weight_dt = DataType["UINT4"]

        conv_config = {}
        conv_config["dilations"] = [0, 0]
        conv_config["group"] = group
        conv_config["kernel_shape"] = [kernel_size, kernel_size]
        conv_config["pads"] = [pad, pad, pad, pad]
        conv_config["strides"] = [stride, stride]

        top_in = helper.make_tensor_value_info("inp", TensorProto.FLOAT, input_shape)
        top_out = helper.make_tensor_value_info("outp", TensorProto.FLOAT, output_shape)
        value_info = [helper.make_tensor_value_info("p1", TensorProto.FLOAT, conv_param_shape)]

        modelproto = qonnx_make_model(
            helper.make_graph(
                name="conv_test",
                inputs=[top_in],
                outputs=[top_out],
                value_info=value_info,
                nodes=[helper.make_node("Conv", ["inp", "p1"], ["outp"], **conv_config)],
            )
        )

        model = ModelWrapper(modelproto)
        model.set_tensor_datatype("inp", idt)
        model.set_tensor_datatype("outp", idt)
        model.set_tensor_datatype("p1", conv_weight_dt)
        model.set_initializer("p1", gen_finn_dt_tensor(conv_weight_dt, conv_param_shape))

        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        new_model = model.transform(LowerConvsToMatMul())
        new_model = new_model.transform(to_hw.InferConvInpGen())
        if not use_rtl_swg:
            for node in new_model.graph.node:
                if is_fpgadataflow_node(node):
                    inst = getCustomOp(node)
                    inst.set_nodeattr("preferred_impl_style", "hls")
        if depthwise is True:
            new_model = new_model.transform(to_hw.InferVectorVectorActivation())
            new_model = new_model.transform(SpecializeLayers(part))
        else:
            new_model = new_model.transform(to_hw.InferQuantizedMatrixVectorActivation())
            new_model = new_model.transform(SpecializeLayers(part))
            # set folding parameters for MVAU
            if new_model.get_nodes_by_op_type("MVAU_hls"):
                fc_node = new_model.get_nodes_by_op_type("MVAU_hls")[0]
            else:
                fc_node = new_model.get_nodes_by_op_type("MVAU_rtl")[0]
            fc_inst = getCustomOp(fc_node)
            mw = fc_inst.get_nodeattr("MW")
            mh = fc_inst.get_nodeattr("MH")
            pe_cands = list(filter(lambda x: mh % x == 0, range(2, mh + 1)))
            simd_cands = list(filter(lambda x: mw % x == 0, range(2, mw + 1)))
            fc_inst.set_nodeattr("PE", pe_cands[0])
            fc_inst.set_nodeattr("SIMD", simd_cands[0])

        new_model = new_model.transform(GiveUniqueNodeNames())
        new_model = new_model.transform(InferShapes())
        new_model = new_model.transform(InferDataTypes())

        # model = model.transform(GiveUniqueNodeNames())
        parent_model = new_model.transform(CreateDataflowPartition())
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    elif model_type == "threshold-only":
        # synthethic MVAU-only model
        mem_mode = "internal_embedded"

        ich = 16
        nf = 1
        act = DataType["INT4"]
        idt = DataType["INT16"]
        # model = make_multi_fclayer_model(128,idt, wdt, adt, 1, impl_style)

        if impl_style == "rtl" and mem_mode == "internal_decoupled":
            pytest.skip(
                "Skip, because test is identical to impl_style=rtl and mem_mode=internal_embedded"
            )
        if nf == -1:
            nf = ich
        pe = ich // nf
        n_inp_vecs = [1, 2, 2]
        assert ich % pe == 0

        # generate input data, data layout is NHWC for FINN
        x = gen_finn_dt_tensor(idt, tuple(n_inp_vecs + [ich]))

        odt = act
        n_steps = act.get_num_possible_values() - 1

        # Generate random, non-decreasing thresholds
        thresholds = generate_random_threshold_values(idt, ich, n_steps)

        thresholds = sort_thresholds_increasing(thresholds)

        if odt == DataType["BIPOLAR"]:
            actval = 0
        else:
            actval = odt.min()

        # Build DUT
        model = make_single_thresholding_modelwrapper(
            impl_style, thresholds, idt, odt, actval, n_inp_vecs
        )

        # Expected Reference output
        # multithreshold util fxn wants NCHW input, not NHWC
        x_nchw = layout_FINN2NCHW(x)
        y = multithreshold(x_nchw, thresholds)

        # convert back to NHWC for comparison to hw outputs
        y = layout_NCHW2FINN(y)
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        else:
            # signed offset
            y += act.min()

        oshape = model.get_tensor_shape("outp")
        y_expected = y.reshape(oshape)

        # package input data as dictionary
        input_dict = {"inp": x}

        # execute DUT
        y_produced = oxe.execute_onnx(model, input_dict)["outp"]

        y_produced = y_produced.reshape(y_expected.shape)

        assert (y_produced == y_expected).all()

        model = model.transform(SpecializeLayers(part))
        # Make sure that SpecializeLayers did not default to
        # HLS implementation unexpectedly
        assert model.graph.node[0].op_type == "Thresholding_" + str(impl_style)
        node = model.graph.node[0]
        inst = getCustomOp(node)
        inst.set_nodeattr("PE", pe)
        if impl_style == "hls":
            inst.set_nodeattr("mem_mode", mem_mode)

        model = model.transform(GiveUniqueNodeNames())
        parent_model = model.transform(CreateDataflowPartition())
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    elif model_type == "mvau-only":
        # synthethic MVAU-only model
        model = make_multi_fclayer_model(128, idt, wdt, adt, 2, exec_mode)
        model = model.transform(GiveUniqueNodeNames())
        parent_model = model.transform(CreateDataflowPartition())
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    elif model_type == "cybersecurity":
        # end2end model cybersecurity

        model_file = "notebooks/end2end_example/cybersecurity/output_estimates_only/intermediate_models/step_generate_estimate_reports.onnx"
        dataflow_model = ModelWrapper(model_file)
    elif model_type == "cnv":
        dataflow_model = ModelWrapper(build_dir + "/../projects/finn/finn-examples/build/bnn-pynq/output_cnv-w2a2_Pynq-Z1_default/intermediate_models/step_create_dataflow_partition.onnx")
        # update datatypes based on the model
        idt = DataType[getCustomOp(dataflow_model.graph.node[0]).get_nodeattr("inputDataType")]
        wdt = DataType[getCustomOp(dataflow_model.graph.node[0]).get_nodeattr("weightDataType")]


    dataflow_model = dataflow_model.transform(SpecializeLayers(part))
    dataflow_model = dataflow_model.transform(AnnotateCycles())

    clk_ns = 1.66

    target_cycles_per_frame = int((10**9 / clk_ns) / target_fps)

    dataflow_model_padded = copy.deepcopy(dataflow_model)
    dataflow_model_naive = dataflow_model.transform(
        SetFolding(target_cycles_per_frame, platform=platform, style="naive")
    )
    dataflow_model_padded = dataflow_model_padded.transform(
        SetFolding(target_cycles_per_frame, platform=platform, style="optimizer")
    )

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
    # assert True == False
    # preparing two identical, sensible thresholding inputs
    # with padding applied to the second

    if model_type == "mvau-only":
        T_naive_list, T_padded_list = generate_thresholding_inputs(
            idt, wdt, dataflow_model_naive, dataflow_model_padded
        )
        dataflow_model_naive = apply_new_thresholds(dataflow_model_naive, T_naive_list)
        dataflow_model_padded = apply_new_thresholds(dataflow_model_padded, T_padded_list)

    input_mw_naive = getCustomOp(dataflow_model_naive.graph.node[0]).get_normal_input_shape()
    output_mh_naive = getCustomOp(dataflow_model_naive.graph.node[-1]).get_normal_output_shape()

    input_mw_padded = getCustomOp(dataflow_model_padded.graph.node[0]).get_normal_input_shape()
    output_mh_padded = getCustomOp(dataflow_model_padded.graph.node[-1]).get_normal_output_shape()
    x = np.zeros(input_mw_padded, dtype=np.float32)
    y = np.zeros(output_mh_padded, dtype=np.float32)

    input_name = dataflow_model_naive.graph.input[0].name
    output_name = dataflow_model_naive.graph.output[0].name

    if exec_mode == "cppsim":
        # dataflow_model_padded.set_initializer(input_name, x)
        #  dataflow_model_padded.set_initializer(output_name, y)

        dataflow_model_naive = update_model(dataflow_model_naive, part)
        dataflow_model_padded = update_model(dataflow_model_padded, part)

        dataflow_model_naive = dataflow_model_naive.transform(SpecializeLayers(part))
        dataflow_model_naive = dataflow_model_naive.transform(GiveUniqueNodeNames())
        dataflow_model_naive = dataflow_model_naive.transform(AnnotateCycles())
        dataflow_model_naive = dataflow_model_naive.transform(SetExecMode("cppsim"))
        dataflow_model_naive = dataflow_model_naive.transform(PrepareCppSim())
        dataflow_model_naive = dataflow_model_naive.transform(CompileCppSim())

        dataflow_model_padded = dataflow_model_padded.transform(SpecializeLayers(part))
        dataflow_model_padded = dataflow_model_padded.transform(GiveUniqueNodeNames())
        dataflow_model_padded = dataflow_model_padded.transform(AnnotateCycles())
        dataflow_model_padded = dataflow_model_padded.transform(SetExecMode("cppsim"))
        dataflow_model_padded = dataflow_model_padded.transform(PrepareCppSim())
        dataflow_model_padded = dataflow_model_padded.transform(CompileCppSim(), cleanup=True)

    elif exec_mode == "rtlsim":
        # input_name = "inp"
        # output_name = "outp"

        # dataflow_model_padded.set_initializer(input_name, x)
        # dataflow_model_padded.set_initializer(output_name, y)

        dataflow_model_naive = update_model(dataflow_model_naive, part)
        dataflow_model_padded = update_model(dataflow_model_padded, part)

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
        dataflow_model_padded = dataflow_model_padded.transform(PrepareRTLSim(), cleanup=True)

    elif exec_mode == "stitched_rtlsim":
        #   input_name = "inp"
        #  output_name = "outp"

        dataflow_model_naive = update_model(dataflow_model_naive, part)

        # v assert True==False

      #  dataflow_model_naive = dataflow_model_naive.transform(SpecializeLayers(part))
     #   dataflow_model_naive = dataflow_model_naive.transform(GiveUniqueNodeNames())

        if len(dataflow_model_naive.graph.input) != 0:
            dataflow_model_naive.graph.input.remove(dataflow_model_naive.graph.input[0])
        input_x = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [*input_mw_naive])
        dataflow_model_naive.graph.input.append(input_x)

        # if (input_mw_padded != input_mw_naive):in
        #     dataflow_model_padded.set_initializer(input_name, x)

        # if (output_mh_padded != output_mh_naive):
        if len(dataflow_model_naive.graph.output) != 0:
            dataflow_model_naive.graph.output.remove(dataflow_model_naive.graph.output[0])
        output_y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [*output_mh_naive])
        dataflow_model_naive.graph.output.append(output_y)

        dataflow_model_naive = dataflow_model_naive.transform(SetExecMode("rtlsim"))
         
        dataflow_model_naive = dataflow_model_naive.transform(PrepareIP(part, clk_ns))
        dataflow_model_naive = dataflow_model_naive.transform(HLSSynthIP())
        dataflow_model_naive = dataflow_model_naive.transform(PrepareRTLSim())

        dataflow_model_naive = dataflow_model_naive.transform(InsertAndSetFIFODepths(part, clk_ns))
        dataflow_model_naive = dataflow_model_naive.transform(PrepareIP(part, clk_ns))
        dataflow_model_naive = dataflow_model_naive.transform(HLSSynthIP())
        dataflow_model_naive = dataflow_model_naive.transform(
            CreateStitchedIP(part, clk_ns), cleanup=True
        )
        dataflow_model_naive.set_metadata_prop("rtlsim_so", "")
        dataflow_model_naive.set_metadata_prop("exec_mode", "rtlsim")

        dataflow_model_padded = update_model(dataflow_model_padded, part)

       # dataflow_model_padded = dataflow_model_padded.transform(SpecializeLayers(part))
       # dataflow_model_padded = dataflow_model_padded.transform(GiveUniqueNodeNames())

        # if (input_mw_padded != input_mw_naive):
        if len(dataflow_model_padded.graph.input) != 0:
            dataflow_model_padded.graph.input.remove(dataflow_model_padded.graph.input[0])

        dataflow_model_padded = dataflow_model_padded.transform(SetExecMode("rtlsim"))
        dataflow_model_padded = dataflow_model_padded.transform(PrepareIP(part, clk_ns))
        dataflow_model_padded = dataflow_model_padded.transform(HLSSynthIP())

        input_x = helper.make_tensor_value_info(
            dataflow_model.graph.node[0].input[0], TensorProto.FLOAT, [*input_mw_padded]
        )
        dataflow_model_padded.graph.input.append(input_x)

        # if (input_mw_padded != input_mw_naive):
        #     dataflow_model_padded.set_initializer(input_name, x)

        # if (output_mh_padded != output_mh_naive):
        if len(dataflow_model_padded.graph.output) != 0:
            dataflow_model_padded.graph.output.remove(dataflow_model_padded.graph.output[0])
        output_y = helper.make_tensor_value_info(
            output_name, TensorProto.FLOAT, [*output_mh_padded]
        )
        dataflow_model_padded.graph.output.append(output_y)

        if model_type == "mvau-only":
            dataflow_model_padded = dataflow_model_padded.transform(InsertAndSetFIFODepths(part, clk_ns))
        dataflow_model_padded = dataflow_model_padded.transform(PrepareIP(part, clk_ns))
        dataflow_model_padded = dataflow_model_padded.transform(HLSSynthIP())
        dataflow_model_padded = dataflow_model_padded.transform(PrepareRTLSim())
        dataflow_model_padded = dataflow_model_padded.transform(
            CreateStitchedIP(part, clk_ns), cleanup=True
        )

        dataflow_model_padded.set_metadata_prop("rtlsim_so", "")
        dataflow_model_padded.set_metadata_prop("exec_mode", "rtlsim")

    # preparing two identical, sensible input vectors with
    # padding applied to the second

    x_input_padded = gen_finn_dt_tensor(idt, input_mw_padded)
    input_dict_padded = prepare_inputs(
        x_input_padded, idt, wdt, inp_name=dataflow_model_padded.graph.node[0].input[0]
    )

    x_input_naive = gen_finn_dt_tensor(idt, input_mw_naive)
    x_input_naive[..., : input_mw_naive[-1]] = x_input_padded[..., : input_mw_naive[-1]]
    input_dict_naive = prepare_inputs(
        x_input_naive, idt, wdt, inp_name=dataflow_model_naive.graph.node[0].input[0]
    )

    y_naive = oxe.execute_onnx(dataflow_model_naive, input_dict_naive)
    y_naive = y_naive[dataflow_model_naive.graph.output[0].name]
    y_naive = y_naive.reshape(
        dataflow_model_naive.get_tensor_shape(dataflow_model_naive.graph.node[-1].output[0])
    )

    y_padded = oxe.execute_onnx(dataflow_model_padded, input_dict_padded)
    y_padded = y_padded[dataflow_model_padded.graph.output[0].name]
    y_padded = y_padded.reshape(
        dataflow_model_padded.get_tensor_shape(dataflow_model_padded.graph.node[-1].output[0])
    )

    y_padded = y_padded[..., : output_mh_naive[-1]]

    assert np.array_equal(y_naive, y_padded)
