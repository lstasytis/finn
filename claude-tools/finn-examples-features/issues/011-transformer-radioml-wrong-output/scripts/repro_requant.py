# Issue 011 minimal reproducer: Requant with an unsigned (UINT8) input, scalar scale
# 0.5, INT8 output, specialized for ZCU104, node rtlsim vs Python execution.
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

part, ch = "xczu7ev-ffvc1156-2-e", 8
inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch])
outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch])
node = helper.make_node(
    "Requant", ["inp", "scale", "bias"], ["outp"], domain="finn.custom_op.fpgadataflow",
    backend="fpgadataflow", NumChannels=ch, PE=1, inputDataType="UINT8",
    outputDataType="INT8", numInputVectors=[1],
)
g = helper.make_graph([node], "g", [inp], [outp])
model = ModelWrapper(qonnx_make_model(g))
model.set_initializer("scale", np.array([0.5], dtype=np.float32))
model.set_initializer("bias", np.array([0.00015259], dtype=np.float32))  # as in transformer_radioml
model.set_tensor_datatype("inp", DataType["UINT8"])
model.set_tensor_datatype("outp", DataType["INT8"])
x = np.array([[0, 1, 100, 127, 128, 192, 200, 254]], dtype=np.float32)
ref = oxe.execute_onnx(model, {"inp": x})["outp"]
model = model.transform(SpecializeLayers(part)).transform(GiveUniqueNodeNames())
print("specialized to", model.graph.node[0].op_type)
model = model.transform(SetExecMode("rtlsim")).transform(PrepareIP(part, 5))
model = model.transform(HLSSynthIP()).transform(PrepareRTLSim())
y = oxe.execute_onnx(model, {"inp": x})["outp"]
print("input ", x.flatten())
print("python", ref.flatten())
print("rtlsim", y.flatten())
print("MATCH" if np.array_equal(ref, y) else "MISMATCH")
