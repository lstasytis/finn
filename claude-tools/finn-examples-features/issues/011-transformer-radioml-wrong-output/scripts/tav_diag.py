# Issue 011: redo the heuristic_analytical TAV derivation of a build (input: its
# step_transpose_decomposition.onnx) and print per-node TAV latencies and the
# derived depths of the edges between two named nodes' neighbourhood.
import sys
from functools import partial

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveTokenAccessVectors,
    derive_chained_tav_depths,
    read_tav_rows,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

part, clk = "xczu7ev-ffvc1156-2-e", 10.0
model = ModelWrapper(sys.argv[1])
model = model.transform(InsertDWC()).transform(SpecializeLayers(part))
model = model.transform(GiveUniqueNodeNames()).transform(AnnotateCycles())
period = int(model.analysis(dataflow_performance)["max_cycles"])
model = model.transform(DeriveTokenAccessVectors(model, period, "tree_model", part, clk))
period = int(model.analysis(partial(dataflow_performance, use_characterized=True))["max_cycles"])
print("period", period)
d_c = derive_chained_tav_depths(model, period, causal=True)
d_p = derive_chained_tav_depths(model, period, causal=False)
a, b = sys.argv[2], sys.argv[3]
names = [n.name for n in model.graph.node]
for n in model.graph.node[names.index(a) : names.index(b) + 1]:
    inst = getCustomOp(n)
    ri, ro = read_tav_rows(inst)

    def first(r):
        if r is None:
            return None
        r = r[0]
        nz = np.nonzero(r)[0]
        return int(nz[0]) if len(nz) else None

    lat = None
    if ri is not None and ro is not None:
        lat = first(ro) - first(ri)
        tok_in_before_out = int(ri[0][first(ro)])
    else:
        tok_in_before_out = None
    print(
        "%-36s TAV first-out - first-in %s cycles, inputs read before first output %s, "
        "exp_cycles %s; out-edge depth causal %s phased %s"
        % (n.name, lat, tok_in_before_out, inst.get_exp_cycles(),
           [d_c.get(t) for t in n.output], [d_p.get(t) for t in n.output])
    )
