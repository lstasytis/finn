"""MLO build (ZU3, no bitfile) of the finnloop end2end model: FINNLoop get_exp_cycles
(old and new formula) vs node rtlsim cycles and vs stitched-IP rtlsim throughput."""
import json
import os

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.basic import gen_finn_dt_tensor

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import make_build_dir
from fpgadataflow.test_fpgadataflow_finnloop import (
    create_chained_loop_bodies,
    create_node,
    create_tensor_info,
    verif_steps,
)

board, dim, simd, pe, mvau_th, helper_pe, bitwidth, weight_bitwidth = (
    "AUP-ZU3_8GB", 12, 3, 6, 1, 6, 8, 8)
iteration = 3
data_dtype = DataType[f"INT{bitwidth}"]
add_out_dtype = DataType[f"INT{bitwidth + 1}"]
bodies = create_chained_loop_bodies(dim, dim, iteration, "ElementwiseAdd_hls", [1],
    data_dtype.name, dtype=data_dtype, mvau_simd=simd, mvau_pe=pe, mvau_th=mvau_th,
    helper_pe=helper_pe, weight_bitwidth=weight_bitwidth)
nodes_per_body = len(bodies[0].graph.node)
model = bodies[0]
for m in bodies[1:]:
    model = model.transform(MergeONNXModels(m))
tail_outp = create_tensor_info("tail_outp", [1, 3, 3, dim])
tr_node = create_node("ElementwiseAdd_hls", [model.graph.output[0].name, "tail_add"],
    ["tail_outp"], "Add_tail", {"lhs_shape": [1, 3, 3, dim], "rhs_shape": [1],
    "out_shape": [1, 3, 3, dim], "lhs_dtype": data_dtype.name, "rhs_dtype": data_dtype.name,
    "out_dtype": add_out_dtype.name})
model.graph.node.insert(len(model.graph.node), tr_node)
model.graph.value_info.append(model.graph.output[0])
model.graph.output.pop(0)
model.graph.output.append(tail_outp)
model.set_initializer("tail_add", gen_finn_dt_tensor(data_dtype, [1]))
model.set_tensor_datatype("tail_add", data_dtype)
model = model.transform(RemoveUnusedTensors())
model = model.transform(InferShapes())
model = model.transform(InferDataTypes())
x = gen_finn_dt_tensor(data_dtype, (1, 3, 3, dim))
ref = model.transform(PrepareCppSim()).transform(CompileCppSim()).transform(SetExecMode("cppsim"))
y_ref = oxe.execute_onnx(ref, {ref.graph.input[0].name: x})[ref.graph.output[0].name]
out = make_build_dir("i009_mlo_")
np.save(out + "/input.npy", x)
np.save(out + "/expected_output.npy", y_ref)
model.save(out + "/mlo_model.onnx")
cfg = build_cfg.DataflowBuildConfig(
    output_dir=out,
    steps=["phase_convert_to_hardware", "phase_optimize_hardware", "phase_build_hardware",
           "phase_generate_outputs"],
    synth_clk_period_ns=10.0, board=board, rtlsim_batch_size=100, standalone_thresholds=True,
    mlo=True, loop_body_hierarchy=[["", "layers.0"]],
    loop_body_range=(model.graph.node[0], model.graph.node[nodes_per_body - 1]),
    verify_steps=verif_steps, verify_input_npy=out + "/input.npy",
    verify_expected_output_npy=out + "/expected_output.npy", mute_config_assertions=True,
    generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                      build_cfg.DataflowOutputType.STITCHED_IP],
)
build.build_dataflow_cfg(out + "/mlo_model.onnx", cfg)
print("BUILD_DIR", out, flush=True)
child = ModelWrapper(out + "/intermediate_models/verify_node_by_node_rtlsim.onnx")
loop = child.get_nodes_by_op_type("FINNLoop")[0]
inst = getCustomOp(loop)
in_name = child.graph.input[0].name
oxe.execute_onnx(child, {in_name: x})
cyc = inst.get_nodeattr("cycles_rtlsim")
body = inst.get_nodeattr("body").transform(AnnotateCycles())
perf = body.analysis(dataflow_performance)
it = inst.get_nodeattr("iteration")
lat, thr = perf["critical_path_cycles"], perf["max_cycles"]
print("FINNLOOP iteration", it, "body critical_path_cycles", lat, "body max_cycles", thr,
      "max node", perf["max_cycles_node_name"])
print("FINNLOOP node rtlsim cycles (1 input)", cyc, "new exp", inst.get_exp_cycles(),
      "old exp", lat + (it - 1) * thr + 40 * it, flush=True)
for n in body.graph.node:
    print("  body node", n.name, getCustomOp(n).get_nodeattr("cycles_estimate"))
for f in ["rtlsim_performance.json", "estimate_network_performance.json"]:
    p = out + "/report/" + f
    if os.path.isfile(p):
        print(f, json.load(open(p)), flush=True)
