import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_config import default_build_dataflow_steps
from qonnx.core.datatype import DataType
from finn.util.basic import make_build_dir
import os
import shutil
import subprocess
import numpy as np
from onnx import helper as oh


from finn.util.visualization import showInNetron
import onnx
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import os
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from qonnx.util.config import extract_model_config_to_json
import numpy as np 
from finn.util.basic import decompress_string_to_numpy
from finn.util.test import (
    compare_two_chr_funcs,
    debug_chr_funcs,
    get_characteristic_fnc,
)

from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
import qonnx.custom_op.registry as registry
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

from finn.transformation.fpgadataflow.prepare_ip import PrepareIP, _codegen_single_node
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

# def _codegen_single_node(node, model, fpgapart, clk):
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)

import onnx
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import os
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from qonnx.transformation.general import (
    GiveUniqueNodeNames,
)
from qonnx.transformation.general import ApplyConfig
from qonnx.util.config import extract_model_config_to_json
from qonnx.custom_op.registry import getCustomOp
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic, DeriveFIFOSizes, StretchCharacteristicFunctions

from finn.util.visualization import showInNetron
import onnx
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import os
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from qonnx.transformation.general import (
    ApplyConfig,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
import numpy as np 
from finn.util.basic import decompress_string_to_numpy
from finn.util.test import (
    compare_two_chr_funcs,
    debug_chr_funcs,
    get_characteristic_fnc,
)

from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
import qonnx.custom_op.registry as registry
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

from finn.transformation.fpgadataflow.prepare_ip import PrepareIP, _codegen_single_node
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim

# def _codegen_single_node(node, model, fpgapart, clk):
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)

import onnx
from qonnx.util.basic import qonnx_make_model
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import os

from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic, DeriveFIFOSizes, StretchCharacteristicFunctions
from finn.util.test import prepare_test_model
from finn.util.finn_examples_model_configs import get_model_configs

import pdb 


def get_fifo_table(rtlsim_model, chr_rtlsim_model, chr_analytical_model):
    def get_depths(model,i):
        inst = getCustomOp(model.graph.node[i])
        if not model.graph.node[i].name.startswith("StreamingFIFO"):
            in_depth = inst.get_nodeattr("inFIFODepths")
            out_depth = inst.get_nodeattr("outFIFODepths")
            print(f'node {inst.onnx_node.name} size: {out_depth}')

           # import pdb
           # if "Convolution" in inst.onnx_node.name:
               # breakpoint()
            chr_node = inst.prepare_kwargs_for_characteristic_fx()
            exp_cycles = inst.get_exp_cycles()
            if chr_node is not None:
                total_clocks, in_clocks, _ = chr_node.get_total_cycles(0)
                _, out_clocks, _ = chr_node.get_total_cycles(1)
                
            else:
                in_clocks = -1
                out_clocks = -1
                total_clocks = -1
            
          #  breakpoint()
            return (model.graph.node[i].name, in_depth[0], out_depth[0],total_clocks,  in_clocks, out_clocks, exp_cycles)
        else:
            return None
    
    def get_all_depths(model):
        l = []
        for i in range(len(model.graph.node)):
            depth = get_depths(model,i)
            if depth is not None:
                l.append(depth)
        return l

    rtlsim_list = get_all_depths(rtlsim_model)
    chr_rtlsim_list = get_all_depths(chr_rtlsim_model)
    chr_analytical_list = get_all_depths(chr_analytical_model)
    
    failed_comparisons = 0

    string = "node analysis post-FIFO sizing:\n"
    string += "="*270+"\n"
    string += f'{str("node name"):<40} | {str("rtlsim_large"):<14} | {str("chr_rtl"):<14} | {str("chr_analytical"):<14} | {str("chr-rtlsim depth"):<19} | {str("chr_total_cycles"):<18} | {str("get_exp_cycles()"):<18} | {str("chr_cycles-exp_cycles"):<21} | {str("input_chr_correct"):<18} | {str("output_chr_correct"):<18}\n'
    string += "="*270+"\n"

    rtlsim_size = 0
    chr_rtlsim_size = 0
    chr_analytical_size = 0
    analyze = None
    #breakpoint()
    for i in range(len(chr_rtlsim_list)):
        input_correct = 1
        output_correct = 1
        chr_rtlsim = chr_rtlsim_list[i]
        chr_analytical = chr_analytical_list[i]
        rtlsim = rtlsim_list[i]
        #breakpoint()
       # print("NODE BEING TESTED: ", chr_rtlsim_list[i][0])
        allowed_chr_offset_positions = 10
        rtlsim = getCustomOp(chr_rtlsim_model.graph.node[i])
        analytical = getCustomOp(chr_analytical_model.graph.node[i])
        #breakpoint()
        period = rtlsim.get_exp_cycles()

        rtlsim.set_nodeattr("io_chrc_period", period)
        

        if not chr_analytical_model.graph.node[i].name.startswith("StreamingFIFO"):
            chr_in = decompress_string_to_numpy(analytical.get_nodeattr("io_chrc_in"))
            chr_out = decompress_string_to_numpy(analytical.get_nodeattr("io_chrc_out"))

            rtlsim_in = decompress_string_to_numpy(rtlsim.get_nodeattr("io_chrc_in"))
            rtlsim_out = decompress_string_to_numpy(rtlsim.get_nodeattr("io_chrc_out"))

            # if chr_rtlsim_list[i][0] == "MVAU_rtl_0":
            #     debug_chr_funcs(chr_in, chr_out, rtlsim_in, rtlsim_out, "input",printout_limit=period)
            #     debug_chr_funcs(chr_in, chr_out, rtlsim_in, rtlsim_out, "output",printout_limit=period)

            try:
                assert compare_two_chr_funcs(
                    chr_in,
                    rtlsim_in,
                    allowed_chr_offset_positions,
                    period_override=period
                )
            except:
               # print("FAILED INPUT CHR COMPARISON")
                failed_comparisons +=1
                input_correct = ""


            try:
                assert compare_two_chr_funcs(
                    chr_out,
                    rtlsim_out,
                    allowed_chr_offset_positions,
                    period_override=period
                )
            except:
               # print("FAILED OUTPUT CHR COMPARISON")
                failed_comparisons +=1
                output_correct = ""


        if i < len(chr_rtlsim_list)-1:
            rtlsim_large_depth_in_next = rtlsim_list[i+1][1]
            chr_old_depth_in_next = chr_rtlsim_list[i+1][1]
            chr_new_depth_in_next = chr_analytical_list[i+1][1]
        else:
            rtlsim_large_depth_in_next = rtlsim_list[i][2]
            chr_old_depth_in_next = chr_rtlsim_list[i][2]
            chr_new_depth_in_next = chr_analytical_list[i][2]

        rtlsim_large_depth_out = rtlsim_list[i][2]
        rtlsim_depth = max(rtlsim_large_depth_out,rtlsim_large_depth_in_next)


        chr_old_depth_out = chr_rtlsim_list[i][2]
        chr_old_depth = max(chr_old_depth_out,chr_old_depth_in_next)

        chr_new_depth_out = chr_analytical_list[i][2]
        chr_new_depth = max(chr_new_depth_out,chr_new_depth_in_next)

        string += f'{str(chr_rtlsim_list[i][0] or "N/A"):<40} | {str(rtlsim_depth):<14} | {str(chr_old_depth):<14} | {str(chr_new_depth):<14} | {str(chr_new_depth-chr_old_depth):<19} | {str(chr_analytical_list[i][3]):<18} | {str(chr_analytical_list[i][6]):<18} | {str(chr_analytical_list[i][3] - chr_analytical_list[i][6]):<21} | {str(input_correct):<18} | {str(output_correct):<18} \n'
        rtlsim_size += rtlsim_list[i][2]
        chr_rtlsim_size += chr_old_depth
        chr_analytical_size += chr_new_depth
        if chr_rtlsim_list[i][0] == "Thresholding_rtl_0":
            analyze2 = i
        if chr_rtlsim_list[i][0] == "Thresholding_rtl_8":
            analyze = i

    string += "="*270+"\n"
    string += f'{str("Sum"):<40} | {str(rtlsim_size):<14} | {str(chr_rtlsim_size-30):<14} | {str(chr_analytical_size-30):<14} | {chr_analytical_size-chr_rtlsim_size}\n'
    string += "="*270+"\n"
    string += f"Characterizations where analytical==rtlsim: {(len(chr_rtlsim_list)*2)-failed_comparisons}/{(len(chr_rtlsim_list)*2)}\n"
    return string, analyze


zynq_platforms = ["ZCU104", "ZCU102", "Pynq-Z1"]
alveo_platforms = ["U250"]
#models_to_test = ["bnn-pynq", "gtsrb", "vgg10-radioml", "cybersecurity-mlp","kws", "mobilenet-v1","resnet50"]
models_to_test = ["kws","cybersecurity-mlp"]
#models_to_test = ["bnn-pynq", "gtsrb", "cybersecurity-mlp","kws"]
#models_to_test = ["mobilenet-v1"]
#models_to_test = ["gtsrb"]
#models_to_test = ["vgg10-radioml"]
#strategies_to_test = [None,  "characterize_analytical", "characterize_rtlsim", "largefifo_rtlsim"]
strategies_to_test = ["largefifo_rtlsim","characterize_analytical"]
#strategies_to_test = ["characterize_rtlsim"]
model_configs = get_model_configs()

model_types = []
dumps = []

runtime_test = False

for model_name, model_config in model_configs.items():
    print(model_config)
    if model_config["model_name"] in models_to_test:
        models = []
        for fifo_sizing_strategy in strategies_to_test:
        
            build_dir = os.environ["FINN_BUILD_DIR"]
            model_root = "finn_examples_models"

            # fetch or generate all necessary models
            model = prepare_test_model(build_dir, model_root, fifo_sizing_strategy, model_config,runtime_test)
            
            #breakpoint()
            models.append(model)
            print(f"added {fifo_sizing_strategy} strategy for {model_config['model_name']} model of config {model_name}")
        model_types.append(models)


        name_of_model = model_name

        part = "xcku3p-ffva676-1-e"
        clk_ns = 10.0

        root_dir = f'{os.getcwd()}'
        rtlsim_model_original = models[0]
        chr_rtlsim_model_original = models[1]
        chr_analytical_model_original = models[2] 
        
        


        # redo analytical characterization
        # this is for testing purposes to check if changes had an effect

        # chr_analytical_model_original = chr_analytical_model_original.transform(AnnotateCycles())
        # period = int(chr_analytical_model_original.analysis(dataflow_performance)["max_cycles"] + 12)
        # chr_analytical_model_original = chr_analytical_model_original.transform(
        #     DeriveCharacteristic(
        #         chr_analytical_model_original,
        #         period,
        #         "analytical",
        #         part,
        #         clk_ns,
        #     )
        # )

        # chr_analytical_model_original = chr_analytical_model_original.transform(StretchCharacteristicFunctions(1,period))
        # chr_analytical_model_original = chr_analytical_model_original.transform(DeriveFIFOSizes(io_fifo_depth=2, period=period))




        dump, analyze = get_fifo_table(rtlsim_model_original, chr_rtlsim_model_original, chr_analytical_model_original)
        print("model: ", model_config["model_name"])
        dumps.append((model_config["model_name"],dump, model_config))


    else:
        print(f"excluded model {model_name}")





for (model_name, dump, cfg) in dumps:
    print("model: ",model_name)
    print(dump)



# second test = runtimes
runtime_test = True
runtime_models = []

for model_name, model_config in model_configs.items():
    print(model_config)
    if model_config["model_name"] in models_to_test:
        models = []
        for fifo_sizing_strategy in strategies_to_test:
        
            build_dir = os.environ["FINN_BUILD_DIR"]
            model_root = "finn_examples_models"

            # fetch or generate all necessary models
            model = prepare_test_model(build_dir, model_root, fifo_sizing_strategy, model_config,runtime_test)
            
            #breakpoint()
            models.append(model)
            print(f"added {fifo_sizing_strategy} strategy for {model_config['model_name']} model of config {model_name}")
        runtime_models.append(models)