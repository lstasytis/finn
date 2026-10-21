"""Issue 015: per-tensor (need, arrival time) of join_window_floors' walk on a build, printed
for the first token, the last of frame 1 and the last of frame 2 (to see where a floor comes
from). usage: floor_debug.py <build dir> <join name> [part]"""
import inspect
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.fpgadataflow import is_hls_node, is_rtl_node

src = inspect.getsource(dc.join_window_floors)
src = src.replace("        memo = {}\n", "        memo = {}\n        MEMOS[join.name] = memo\n")
ns = dict(vars(dc))
ns["MEMOS"] = MEMOS = {}
exec(src, ns)
m = ModelWrapper(sys.argv[1] + "/intermediate_models/step_transpose_decomposition.onnx")
m = m.transform(InsertDWC()).transform(SpecializeLayers(sys.argv[3] if len(sys.argv) > 3 else "xczu7ev-ffvc1156-2-e"))
m = m.transform(GiveUniqueNodeNames())
hw = [n for n in m.graph.node if is_hls_node(n) or is_rtl_node(n)]
print(ns["join_window_floors"](m, hw))
order = {n.name: i for i, n in enumerate(m.graph.node)}
memo = MEMOS[sys.argv[2]]
for t in sorted(memo, key=lambda t: order[m.find_producer(t).name]):
    need, times = memo[t]
    p = m.find_producer(t)
    from qonnx.custom_op.registry import getCustomOp
    inst = getCustomOp(p)
    o = list(p.output).index(t)
    n = dc._frame_words(inst, o, output=True)
    idx = [0, min(n, len(times)) - 1, len(times) - 1]
    try:
        ec = inst.get_exp_cycles()
    except Exception:
        ec = None
    print(f"{p.name:34s} n/frame {n:6d} exp {ec} need {[int(need[i]) for i in idx]} "
          f"t {[int(times[i]) for i in idx]}")
