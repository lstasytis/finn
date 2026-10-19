import sys, os
sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "fpgadataflow"))
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from test_fifosizing import make_residual_modelwrapper

def residual_mvau(nvec=16, ch=64, simd=8, pe=64, idt=DataType["INT4"]):
    shape = [1, nvec, ch]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, shape)
    vi = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in ("short", "long0", "long1")]
    nodes = [
        helper.make_node("DuplicateStreams", ["inp"], ["short", "long0"], domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
            NumChannels=ch, NumOutputStreams=2, PE=simd, inputDataType=idt.name, numInputVectors=[1, nvec], outFIFODepths=[2, 2]),
        helper.make_node("MVAU", ["long0", "w"], ["long1"], domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
            MW=ch, MH=ch, SIMD=simd, PE=pe, inputDataType=idt.name, weightDataType=idt.name, outputDataType=idt.name,
            ActVal=0, binaryXnorMode=0, noActivation=1, numInputVectors=[1, nvec], mem_mode="internal_decoupled", resType="lut"),
        helper.make_node("ElementwiseAdd", ["short", "long1"], ["outp"], domain="finn.custom_op.fpgadataflow", backend="fpgadataflow",
            lhs_shape=shape, rhs_shape=shape, out_shape=shape, lhs_dtype=idt.name, rhs_dtype=idt.name, out_dtype=idt.name,
            lhs_style="input", rhs_style="input", PE=pe, inFIFODepths=[2, 2]),
    ]
    graph = helper.make_graph(nodes, "residual_mvau", [inp], [outp], value_info=vi)
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w", gen_finn_dt_tensor(idt, (ch, ch)))
    model.set_tensor_datatype("w", idt)
    for t in ["inp", "outp", "short", "long0", "long1"]:
        model.set_tensor_datatype(t, idt)
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.derive_characteristic import DeriveTokenAccessVectors
import importlib.util
_spec=importlib.util.spec_from_file_location('dcv', sys.argv[1] if len(sys.argv)>1 else os.environ['FINN_ROOT']+'/src/finn/transformation/fpgadataflow/derive_characteristic.py'); dcv=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dcv); DeriveFIFOSizes=dcv.DeriveFIFOSizes
from functools import partial

def mlp_model(layers, nvec=1, idt=DataType["INT4"], wdt=DataType["INT4"]):
    """layers: list of (mw, mh, simd, pe)."""
    nodes, vi = [], []
    cur = "inp"
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [nvec, layers[0][0]])
    for i, (mw, mh, simd, pe) in enumerate(layers):
        out = "outp" if i == len(layers) - 1 else "t%d" % i
        nodes.append(helper.make_node("MVAU", [cur, "w%d" % i], [out], domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow", MW=mw, MH=mh, SIMD=simd, PE=pe, inputDataType=idt.name,
            weightDataType=wdt.name, outputDataType=idt.name, ActVal=0, binaryXnorMode=0, noActivation=1,
            numInputVectors=[nvec], mem_mode="internal_decoupled", resType="lut"))
        if out != "outp":
            vi.append(helper.make_tensor_value_info(out, TensorProto.FLOAT, [nvec, mh]))
        cur = out
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [nvec, layers[-1][1]])
    graph = helper.make_graph(nodes, "mlp", [inp], [outp], value_info=vi)
    model = ModelWrapper(qonnx_make_model(graph))
    for i, (mw, mh, simd, pe) in enumerate(layers):
        model.set_initializer("w%d" % i, gen_finn_dt_tensor(wdt, (mw, mh)))
        model.set_tensor_datatype("w%d" % i, wdt)
    model.set_tensor_datatype("inp", idt)
    for n in ["outp"] + [v.name for v in vi]:
        model.set_tensor_datatype(n, idt)
    model = model.transform(InferShapes())
    return model.transform(InferDataTypes())

def size(model, part="xc7z020clg400-1", clk=10.0):
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(AnnotateCycles())
    period = int(model.analysis(dataflow_performance)["max_cycles"])
    model = model.transform(DeriveTokenAccessVectors(model, period, "tree_model", part, clk))
    period = int(model.analysis(partial(dataflow_performance, use_characterized=True))["max_cycles"])
    model = model.transform(DeriveFIFOSizes(period=period))
    out = {}
    for n in model.graph.node:
        inst = getCustomOp(n)
        for i, t in enumerate(n.output):
            c = model.find_consumer(t)
            if c is not None:
                ci = getCustomOp(c)
                d = max(inst.get_nodeattr("outFIFODepths")[i], ci.get_nodeattr("inFIFODepths")[list(c.input).index(t)])
                out["%s -> %s" % (n.name, c.name)] = d
    return out, period

for name, m in [("mlp", mlp_model([(64, 64, 64, 64), (64, 64, 8, 1), (64, 64, 64, 64), (64, 64, 8, 8)], nvec=16)), ("mlp2", mlp_model([(64, 64, 8, 8), (64, 64, 2, 64), (64, 64, 64, 64)], nvec=16)), ("residual", make_residual_modelwrapper()), ("residual_mvau", residual_mvau())]:
    d, p = size(m)
    print(name, "period", p)
    for k, v in d.items():
        print("  ", k, v)
