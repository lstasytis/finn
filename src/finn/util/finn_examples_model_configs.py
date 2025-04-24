# Custom steps for finn-examples models

import numpy as np
import onnx
import os
import qonnx.custom_op.registry as registry
import shutil
import subprocess
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from qonnx.transformation.change_3d_tensors_to_4d import Change3DTo4DTensors
from qonnx.transformation.change_datalayout import ChangeDataLayoutQuantAvgPool2d
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    ApplyConfig,
    ConvertDivToMul,
    ConvertSubToAdd,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
    SortGraph,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.util.basic import qonnx_make_model
from qonnx.util.config import extract_model_config_to_json

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.reorder as reorder
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    ShellFlowType,
    default_build_dataflow_steps,
)
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveCharacteristic,
    DeriveFIFOSizes,
    StretchCharacteristicFunctions,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP, _codegen_single_node
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

# def _codegen_single_node(node, model, fpgapart, clk):
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.absorb import (
    Absorb1BitMulIntoConv,
    Absorb1BitMulIntoMatMul,
    AbsorbAddIntoMultiThreshold,
    AbsorbConsecutiveTransposes,
    AbsorbMulIntoMultiThreshold,
    AbsorbScalarMulAddIntoTopK,
    AbsorbTransposeIntoMultiThreshold,
    FactorOutMulSignMagnitude,
)
from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul,
)

# just for not linear
from finn.transformation.streamline.reorder import (
    MoveAddPastConv,
    MoveAddPastMul,
    MoveLinearPastEltwiseAdd,
    MoveLinearPastFork,
    MoveMaxPoolPastMultiThreshold,
    MoveScalarAddPastMatMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastConv,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.sign_to_thres import ConvertSignToThres
from finn.util.basic import decompress_string_to_numpy, make_build_dir
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.util.test import (
    compare_two_chr_funcs,
    debug_chr_funcs,
    get_characteristic_fnc,
)
from finn.util.visualization import showInNetron


def step_mobilenet_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Streamline())
    additional_streamline_transformations = [
        DoubleToSingleFloat(),
        reorder.MoveMulPastDWConv(),
        absorb.AbsorbMulIntoMultiThreshold(),
        ChangeDataLayoutQuantAvgPool2d(),
        InferDataLayouts(),
        reorder.MoveTransposePastScalarMul(),
        absorb.AbsorbTransposeIntoFlatten(),
        reorder.MoveFlattenPastAffine(),
        reorder.MoveFlattenPastTopK(),
        reorder.MoveScalarMulPastMatMul(),
        CollapseRepeatedMul(),
        RemoveIdentityOps(),
        RoundAndClipThresholds(),
    ]
    for trn in additional_streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
    return model


def step_mobilenet_lower_convs(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(LowerConvsToMatMul())
    model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferDataLayouts())
    return model


def step_mobilenet_convert_to_hw_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


def step_mobilenet_slr_floorplan(model: ModelWrapper, cfg: DataflowBuildConfig):
    if cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        try:
            from finnexperimental.analysis.partitioning import partition

            # apply partitioning of the model, restricting the first and last layers
            # to SLR0
            default_slr = 0
            abs_anchors = [(0, [default_slr]), (-1, [default_slr])]
            floorplan = partition(
                model,
                cfg.synth_clk_period_ns,
                cfg.board,
                abs_anchors=abs_anchors,
                multivariant=False,
            )[0]
            # apply floorplan to model
            model = model.transform(ApplyConfig(floorplan))
            print("SLR floorplanning applied")
        except Exception:
            print("No SLR floorplanning applied")
    return model


def step_mobilenet_convert_to_hw_layers_separate_th(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferPool())
    model = model.transform(to_hw.InferConvInpGen())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferVectorVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model


from qonnx.transformation.insert_topk import InsertTopK


# Inject the preprocessing step into FINN to enable json serialization later on
def step_preprocess(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(InsertTopK(k=1))
    return model


def step_pre_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(Change3DTo4DTensors())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    return model


def step_convert_final_layers(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(to_hw.InferChannelwiseLinearLayer())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(GiveUniqueNodeNames())
    return model


def custom_step_add_preproc_GTSRB(model, cfg):
    # GTSRB data with raw uint8 pixels is divided by 255 prior to training
    # reflect this in the inference graph so we can perform inference directly
    # on raw uint8 data
    in_name = model.graph.input[0].name
    new_in_name = model.make_new_valueinfo_name()
    new_param_name = model.make_new_valueinfo_name()
    div_param = np.asarray(255.0, dtype=np.float32)
    new_div = oh.make_node(
        "Div",
        [in_name, new_param_name],
        [new_in_name],
        name="PreprocDiv",
    )
    model.set_initializer(new_param_name, div_param)
    model.graph.node.insert(0, new_div)
    model.graph.node[1].input[0] = new_in_name
    # set input dtype to uint8
    model.set_tensor_datatype(in_name, DataType["UINT8"])
    return model


def step_resnet50_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(InsertTopK())
    model = model.transform(InferShapes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model


def step_resnet50_streamline_linear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        AbsorbScalarMulAddIntoTopK(),  # before MoveAddPastMul to avoid int->float
        ConvertSubToAdd(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
        CollapseRepeatedMul(),
        BatchNormToAffine(),
        ConvertSignToThres(),
        MoveAddPastMul(),
        MoveScalarAddPastMatMul(),
        MoveAddPastConv(),
        MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        MoveScalarLinearPastInvariants(),
        MoveAddPastMul(),
        CollapseRepeatedAdd(),
        CollapseRepeatedMul(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        Absorb1BitMulIntoMatMul(),
        Absorb1BitMulIntoConv(),
        RoundAndClipThresholds(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_resnet50_streamline_nonlinear(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        MoveLinearPastEltwiseAdd(),
        MoveLinearPastFork(),
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model


def step_resnet50_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    for iter_id in range(4):
        model = step_resnet50_streamline_linear(model, cfg)
        model = step_resnet50_streamline_nonlinear(model, cfg)

        # big loop tidy up
        model = model.transform(RemoveUnusedTensors())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(SortGraph())

    model = model.transform(DoubleToSingleFloat())

    return model


def step_resnet50_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    to_hw_transformations = [
        to_hw.InferAddStreamsLayer,
        LowerConvsToMatMul,
        to_hw.InferChannelwiseLinearLayer,
        to_hw.InferPool,
        AbsorbTransposeIntoMultiThreshold,
        RoundAndClipThresholds,
        to_hw.InferQuantizedMatrixVectorActivation,
        to_hw.InferThresholdingLayer,
        AbsorbConsecutiveTransposes,
        to_hw.InferConvInpGen,
        to_hw.InferDuplicateStreamsLayer,
        to_hw.InferLabelSelectLayer,
    ]
    for trn in to_hw_transformations:
        model = model.transform(trn())
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())

    return model


def step_resnet50_slr_floorplan(model: ModelWrapper, cfg: DataflowBuildConfig):
    if cfg.shell_flow_type == ShellFlowType.VITIS_ALVEO:
        # previously, we would always ran the finn experimental partitioner on ResNet-50
        # this is now changed and a fixed floorplan is applied
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(ApplyConfig("floorplan_resnet50.json"))
        print("Fixed SLR floorplanning applied")

        # if you would like to try out the experimental partitioner
        # please uncomment the lines (that are not marked as comment) below.

        # import numpy as np
        # from finnexperimental.analysis.partitioning import partition

        # comment: apply partitioning of the model, restricting the first and last layer to SLR0
        # default_slr = 0
        # abs_anchors = [(0, [default_slr]), (-1, [default_slr])]

        # comment: increase resource limits to make partitioning feasible, except for SLR0
        # comment: which also has DDR subsystem
        # limits = np.array(
        #    [
        #        [0.75, 0.5, 0.7, 0.6, 0.6],
        #        [1, 0.7, 0.9, 0.8, 0.8],
        #        [1, 0.7, 0.9, 0.8, 0.8],
        #        [1, 0.7, 0.9, 0.8, 0.8],
        #    ]
        # )
        # floorplan = partition(
        #    model,
        #    cfg.synth_clk_period_ns,
        #    cfg.board,
        #    abs_anchors=abs_anchors,
        #    multivariant=False,
        #    linear_cuts=True,
        #    limits=limits,
        # )[0]

        # comment: apply floorplan to model
        # model = model.transform(ApplyConfig(floorplan))
        # print("SLR floorplanning applied from partitioner")
    return model


def get_model_configs():
    models = {
        "config0": {
            "model_name": "bnn-pynq",
            "model_config": "cnv-w1a1",
            "platform": "Pynq-Z1",
            "dataflow_steps": [
                "step_qonnx_to_finn",
                "step_tidy_up",
                "step_streamline",
                "step_convert_to_hw",
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": "cnv-w1a1_specialize_layers.json",
            "folding_config_json_path": "cnv-w1a1_folding_config.json",
            "fps": None,
            "mvau_wwidth_max": 80,
            "clk_ns": 5.0,
        },
        "config1": {
            "model_name": "bnn-pynq",
            "model_config": "tfc-w1a1",
            "platform": "Pynq-Z1",
            "dataflow_steps": [
                "step_qonnx_to_finn",
                "step_tidy_up",
                "step_streamline",
                "step_convert_to_hw",
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": "tfc-w1a1_specialize_layers.json",
            "folding_config_json_path": "tfc-w1a1_folding_config.json",
            "fps": None,
            "mvau_wwidth_max": 80,
            "clk_ns": 5.0,
        },
        "config2": {
            "model_name": "gtsrb",
            "model_config": "cnv_1w1a_gtsrb",
            "platform": "Pynq-Z1",
            "dataflow_steps": [custom_step_add_preproc_GTSRB]
            + [
                "step_qonnx_to_finn",
                "step_tidy_up",
                "step_streamline",
                "step_convert_to_hw",
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": "gtsrb_specialize_layers.json",
            "folding_config_json_path": "gtsrb_folding_config.json",
            "fps": 3000,
            "mvau_wwidth_max": 36,
            "clk_ns": 10.0,
        },
        "config3": {
            "model_name": "vgg10-radioml",
            "model_config": "radioml_w4a4_small_tidy",
            "platform": "ZCU104",
            "dataflow_steps": [
                "step_tidy_up",
                step_pre_streamline,
                "step_streamline",
                "step_convert_to_hw",
                step_convert_final_layers,
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": "ZCU104_specialize_layers.json",
            "folding_config_json_path": "ZCU104_folding_config.json",
            "fps": None,
            "mvau_wwidth_max": 36,
            "clk_ns": 4.0,
        },
        "config4": {
            "model_name": "kws",
            "model_config": "MLP_W3A3_python_speech_features_pre-processing_QONNX_opset-11",
            "platform": "Pynq-Z1",
            "dataflow_steps": [
                step_preprocess,
                "step_qonnx_to_finn",
                "step_tidy_up",
                "step_streamline",
                "step_convert_to_hw",
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": "kws_specialize_layers.json",
            "folding_config_json_path": "kws_folding_config.json",
            "fps": None,
            "mvau_wwidth_max": 36,
            "clk_ns": 10.0,
        },
        "config5": {
            "model_name": "cybersecurity-mlp",
            "model_config": "unsw_nb15-mlp-w2a2",
            "platform": "Pynq-Z1",
            "dataflow_steps": [
                "step_qonnx_to_finn",
                "step_tidy_up",
                "step_streamline",
                "step_convert_to_hw",
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": None,
            "folding_config_json_path": None,
            "fps": 1000000,
            "mvau_wwidth_max": 80,
            "clk_ns": 10.0,
        },
        "config6": {
            "model_name": "mobilenet-v1",
            "model_config": "mobilenetv1-w4a4_pre_post_tidy_opset-11",
            "platform": "ZCU104",
            "dataflow_steps": [
                step_mobilenet_streamline,
                step_mobilenet_lower_convs,
                step_mobilenet_convert_to_hw_layers_separate_th,
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": "ZCU104_specialize_layers.json",
            "folding_config_json_path": "ZCU104_folding_config.json",
            "fps": None,
            "mvau_wwidth_max": 36,
            "clk_ns": 5.4,
        },
        "config7": {
            "model_name": "resnet50",
            "model_config": "resnet50_w1a2_exported",
            "platform": "U250",
            "dataflow_steps": [
                step_resnet50_tidy,
                step_resnet50_streamline,
                step_resnet50_convert_to_hw,
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
            ],
            "largefifo_rtlsim_depths_json_path": "",
            "characterize_rtlsim_fifo_depths_json_path": "",
            "characterize_analytical_fifo_depths_json_path": "",
            "specialize_layers_json_path": "U250_specialize_layers.json",
            "folding_config_json_path": "U250_folding_config.json",
            "fps": 300,
            "mvau_wwidth_max": 36,
            "clk_ns": 4.0,
        },
    }

    return models
