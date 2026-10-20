# Issue 011 minimal reproducer: GlobalAccPool_hls on a non-square (1 x 8) input,
# as the transformers' mean pooling over tokens produces. cppsim vs numpy sum.
import sys

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

ih, iw = (int(v) for v in sys.argv[1:3]) if len(sys.argv) > 2 else (1, 8)
ch, pe, idt = 8, 2, DataType["UINT4"]
inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ih, iw, ch])
outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, 1, 1, ch])
node = helper.make_node(
    "GlobalAccPool", ["inp"], ["outp"], domain="finn.custom_op.fpgadataflow",
    backend="fpgadataflow", NumChannels=ch, PE=pe, inputDataType=idt.name,
    numInputVectors=[1, ih, iw], preferred_impl_style="hls",
)
model = ModelWrapper(qonnx_make_model(helper.make_graph([node], "g", [inp], [outp])))
model.set_tensor_datatype("inp", idt)
model.set_tensor_datatype("outp", DataType["UINT8"])
model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))
model = model.transform(GiveUniqueNodeNames())
model = model.transform(PrepareCppSim()).transform(CompileCppSim()).transform(SetExecMode("cppsim"))
np.random.seed(0)
x = gen_finn_dt_tensor(idt, (1, ih, iw, ch))
y = oxe.execute_onnx(model, {"inp": x})["outp"].flatten()
print("expected", np.sum(x, axis=(1, 2)).flatten())
print("cppsim  ", y)
print("first pixel only", x[0, 0, 0])
print("MATCH" if (y == np.sum(x, axis=(1, 2)).flatten()).all() else "MISMATCH")
