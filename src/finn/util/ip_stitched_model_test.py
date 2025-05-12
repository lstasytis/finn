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
import json
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


zynq_platforms = ["ZCU104", "ZCU102", "Pynq-Z1"]
alveo_platforms = ["U250"]
#models_to_test = ["bnn-pynq", "gtsrb", "vgg10-radioml", "cybersecurity-mlp","kws", "mobilenet-v1","resnet50"]
#models_to_test = ["bnn-pynq", "gtsrb"]
#models_to_test = ["bnn-pynq", "gtsrb", "vgg10-radioml", "cybersecurity-mlp","kws"]
#models_to_test = ["bnn-pynq", "gtsrb", "cybersecurity-mlp","kws"]
#models_to_test = ["gtsrb"]
models_to_test = ["vgg10-radioml"]
#models_to_test = 
strategies_to_test = ["characterize_analytical", "characterize_rtlsim", "largefifo_rtlsim"]
#strategies_to_test = ["largefifo_rtlsim","characterize_analytical"]
#strategies_to_test = ["largefifo_rtlsim"]
model_configs = get_model_configs()

model_types = []

# second test = runtimes
runtime_test = True
runtime_models = []

results = []

for model_name, model_config in model_configs.items():
    print(model_config)
    if model_config["model_name"] in models_to_test:
        models = []
        for fifo_sizing_strategy in strategies_to_test:
            total_depth = 0
            build_dir = os.environ["FINN_BUILD_DIR"]
            model_root = "finn_examples_models"

            # fetch or generate all necessary models
            
            
            #model = prepare_test_model(build_dir, model_root, fifo_sizing_strategy, model_config,runtime_test)
            model = None
            for x in os.listdir(build_dir):
                if x.startswith(
                    f"build_finn_examples_tests_{model_config['model_name']}_{model_config['model_config']}_{model_config['platform']}_{fifo_sizing_strategy}_"
                ):
                    model_file = f"{build_dir}/{x}/intermediate_models/step_measure_rtlsim_performance.onnx"
                    rtlsim_report = f"{build_dir}/{x}/report/rtlsim_performance.json"
                    if os.path.isfile(model_file):
                        print("Found fully stitched and measured model for this strategy")
                        model = ModelWrapper(model_file)

            if model is not None:

                print(f"adding {fifo_sizing_strategy} strategy for {model_config['model_name']} model of config {model_name}")
                
                for node in model.graph.node:
                    inst = getCustomOp(node)
                    if node.name.startswith("StreamingFIFO"):
                        total_depth += inst.get_nodeattr("depth")
                      #  breakpoint()
                # read json to get perf
                fps = json.load(open(rtlsim_report))["stable_throughput[images/s]"]

                
            models.append([model_config["model_name"], fifo_sizing_strategy, total_depth, fps])
        runtime_models.append(models)

for model in runtime_models:
    for strategy in model:
        print(strategy)