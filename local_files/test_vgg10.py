#the rerun of only the characterization and not the initial building

# store?
#onnx_model1 = qonnx_make_model(model.graph, producer_name="simple-model1")
#onnx.save(onnx_model1, f'/{root_dir}/vg10_model_to_derive.onnx')


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

part = "xcku3p-ffva676-1-e"
clk_ns = 10.0

root_dir = f'{os.getcwd()}'
model = ModelWrapper(f'/{root_dir}/vg10_model_to_derive.onnx')

chr_rtlsim_model_original = ModelWrapper(f'/{root_dir}/vg10_model_to_derive_rtlsim_original.onnx')

# Re-import the class
from finn.transformation.fpgadataflow.derive_characteristic import DeriveCharacteristic, DeriveFIFOSizes


for node in model.graph.node:
    inst = getCustomOp(node)
    inst.set_nodeattr("inFIFODepths",[0])
#    if "Thresh" in inst.onnx_node.name:
  #  print("setting batch size to 1")
    numReps = int(np.prod(inst.get_folded_input_shape()[0:2]))
    print(f'{inst.onnx_node.name} numReps: {numReps}')
      #  inst.set_nodeattr("numInputVectors",[1,1])
    #inst.set_nodeattr("io_chrc_period",[0])

period = int(model.analysis(dataflow_performance)["max_cycles"] + 12)
model = model.transform(
    DeriveCharacteristic(
        model,
        period,
        "analytical",
        part,
        clk_ns,
    )
)

print("Deriving ANALYTICAL chr")
model = model.transform(DeriveFIFOSizes())



chr_rtlsim_model_path = f'/{root_dir}/VGG10_build_output/run_2_characterize_rtlsim/intermediate_models/step_set_fifo_depths.onnx'
rtlsim_model_path = f'/{root_dir}/VGG10_build_output/run_0_largefifo_rtlsim/intermediate_models/step_set_fifo_depths.onnx'

#chr_rtlsim_model_original = ModelWrapper(chr_rtlsim_model_path)
rtlsim_model_original = ModelWrapper(rtlsim_model_path)


print("Deriving rtlsim chr")
chr_rtlsim_model_original = chr_rtlsim_model_original.transform(DeriveFIFOSizes())

def get_fifo_table(rtlsim_model, chr_rtlsim_model, chr_analytical_model):
    def get_depths(model,i):
        inst = getCustomOp(model.graph.node[i])
        if "FIFO" not in model.graph.node[i].name:
            in_depth = inst.get_nodeattr("inFIFODepths")
            out_depth = inst.get_nodeattr("outFIFODepths")
            print(f'node {inst.onnx_node.name} size: {out_depth}')
            return (model.graph.node[i].name,in_depth[0], out_depth[0])
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
    

    string = ""
    # string += "inFIFODepths:\n"
    # string += f'{str("node"):<40} | {str("rtlsim_large"):<13} | {str("chr_rtlsim"):<13} | {str("chr_analytical"):<13}\n'
    
    # for i in range(len(chr_rtlsim_list)):
    #     string += f'{str(chr_rtlsim_list[i][0] or "N/A"):<40} | {str(rtlsim_list[i][1]):<13} | {str(chr_rtlsim_list[i][1]):<13} | {str(chr_analytical_list[i][1]):<13}\n'
    


    
    string += "outFIFODepths:\n"
    string += "=======================================================================================================================\n"
    string += f'{str("node"):<40} | {str("rtlsim_large"):<14} | {str("chr_rtlsim"):<14} | {str("chr_analytical"):<14} | chr_analytical-chr_rtlsim\n'
    string += "=======================================================================================================================\n"

    rtlsim_size = 0
    chr_rtlsim_size = 0
    chr_analytical_size = 0
    analyze = None
    for i in range(len(chr_rtlsim_list)):
        string += f'{str(chr_rtlsim_list[i][0] or "N/A"):<40} | {str(rtlsim_list[i][2]):<14} | {str(chr_rtlsim_list[i][2]):<14} | {str(chr_analytical_list[i][2]):<14} | {chr_analytical_list[i][2]-chr_rtlsim_list[i][2]}\n'
        rtlsim_size += rtlsim_list[i][2]
        chr_rtlsim_size += chr_rtlsim_list[i][2]
        chr_analytical_size += chr_analytical_list[i][2]
        if chr_rtlsim_list[i][0] == "Thresholding_rtl_0":
            analyze2 = i
        if chr_rtlsim_list[i][0] == "MVAU_rtl_0":
            analyze = i

    string += "=======================================================================================================================\n"
    string += f'{str("Sum"):<40} | {str(rtlsim_size):<14} | {str(chr_rtlsim_size):<14} | {str(chr_analytical_size):<14} | {chr_analytical_size-chr_rtlsim_size}\n'
    string += "=======================================================================================================================\n"
    return string, analyze

dump, analyze = get_fifo_table(rtlsim_model_original, chr_rtlsim_model_original, model)
print(dump)



if analyze is not None:
    print("analysis!")
    rtlsim = getCustomOp(chr_rtlsim_model_original.graph.node[analyze])
    analytical = getCustomOp(model.graph.node[analyze])


    chr_in = decompress_string_to_numpy(analytical.get_nodeattr("io_chrc_in"))
    chr_out = decompress_string_to_numpy(analytical.get_nodeattr("io_chrc_out"))

    rtlsim_in = decompress_string_to_numpy(rtlsim.get_nodeattr("io_chrc_in"))
    rtlsim_out = decompress_string_to_numpy(rtlsim.get_nodeattr("io_chrc_out"))

    debug_chr_funcs(chr_in, chr_out, rtlsim_in, rtlsim_out, "input")
    debug_chr_funcs(chr_in, chr_out, rtlsim_in, rtlsim_out, "output")



