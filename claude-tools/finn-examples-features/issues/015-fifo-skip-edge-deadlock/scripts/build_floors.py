"""Issue 015: join floors (join_window_floors) of a build's folding, and the depths its FIFO
sizing gave the same join inputs (from step_set_fifo_depths.onnx: the FIFO in front of the
join input). usage: build_floors.py <build dir> [part]"""
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.transformation.fpgadataflow.derive_characteristic import join_window_floors
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

bdir = sys.argv[1]
part = sys.argv[2] if len(sys.argv) > 2 else "xczu7ev-ffvc1156-2-e"
m = ModelWrapper(bdir + "/intermediate_models/step_transpose_decomposition.onnx")
m = m.transform(InsertDWC()).transform(SpecializeLayers(part)).transform(GiveUniqueNodeNames())
hw = [n for n in m.graph.node if is_hls_node(n) or is_rtl_node(n)]
floors = join_window_floors(m, hw)
# sized depths: FIFO in front of each join input, by (join name, input index)
sized = {}
try:
    f = ModelWrapper(bdir + "/intermediate_models/step_set_fifo_depths.onnx")
    for n in f.graph.node:
        if n.op_type.startswith(("ElementwiseAdd", "AddStreams")):
            for i, t in enumerate(n.input):
                p = f.find_producer(t)
                if p is not None and p.op_type.startswith("StreamingFIFO"):
                    sized[(n.name, i)] = (p.name, getCustomOp(p).get_nodeattr("depth"))
except Exception as e:
    print("no sized model:", e)
for t, fl in floors.items():
    c = m.find_consumer(t)
    i = list(c.input).index(t)
    print(f"{m.find_producer(t).name:28s} -> {c.name:22s} in{i} floor {fl:7d} sized {sized.get((c.name, i))}")
