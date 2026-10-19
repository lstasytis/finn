import copy, sys, time
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.minimize_accumulator_width import MinimizeAccumulatorWidth
from finn.transformation.fpgadataflow.minimize_weight_bit_width import MinimizeWeightBitWidth
from finn.util.basic import part_map
part = part_map[sys.argv[2]]
m = ModelWrapper(sys.argv[1]).transform(GiveUniqueNodeNames()).transform(InsertDWC()).transform(SpecializeLayers(part))
def dump(m):
    d = {}
    for n in m.graph.node:
        i = getCustomOp(n)
        if n.op_type.startswith(("MVAU", "Thresh", "VVAU")):
            t = i.get_nodeattr_types()
            d[n.name] = (i.node_res_estimation(part)["LUT"], i.get_nodeattr("accDataType") if "accDataType" in t else None,
                         i.get_input_datatype(1) if n.op_type.startswith("MVAU") else None,
                         i.get_nodeattr("ram_style_thresholds") if "ram_style_thresholds" in t else None)
    return d
a = dump(m)
t0 = time.time()
m2 = copy.deepcopy(m).transform(MinimizeWeightBitWidth()).transform(MinimizeAccumulatorWidth()).transform(InferDataTypes())
print("as-built transforms", round(time.time() - t0, 2), "s")
b = dump(m2)
for k in a:
    print(f"{k:22s} {a[k][0]:>7} -> {b[k][0]:>7}  acc {a[k][1]} -> {b[k][1]}  w {a[k][2]} -> {b[k][2]} thr {a[k][3]}")
