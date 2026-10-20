"""Run the heuristic_analytical part of step_set_fifo_depths up to DeriveTokenAccessVectors on a
build and save the model with its TAVs (io_chrc_*) and the sizing period, so that
derive_chained_tav_depths can be iterated on offline.
usage: tav_model.py <build dir> <out.onnx>; run from a worktree with wtenv.sh."""
import json
import sys
from functools import partial

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.derive_characteristic import (
    DeriveTokenAccessVectors,
    JustInTimeSynthesize,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

PARTS = {
    "ZCU104": "xczu7ev-ffvc1156-2-e",
    "U55C": "xcu55c-fsvh2892-2L-e",
    "V80": "xcv80-lsva4737-2MHP-e-s",
}
bdir, out = sys.argv[1], sys.argv[2]
part = [v for k, v in PARTS.items() if k in bdir][0]
m = ModelWrapper(bdir + "/intermediate_models/step_transpose_decomposition.onnx")
m = m.transform(InsertDWC())
m = m.transform(SpecializeLayers(part))
m = m.transform(GiveUniqueNodeNames())
m = m.transform(AnnotateCycles())
m = m.transform(JustInTimeSynthesize(part, 10.0, True))
period = int(m.analysis(dataflow_performance)["max_cycles"])
m = m.transform(DeriveTokenAccessVectors(m, period, "tree_model", part, 10.0))
period = int(m.analysis(partial(dataflow_performance, use_characterized=True))["max_cycles"])
m.set_metadata_prop("sizing_period", str(period))
m.save(out)
print("SAVED", out, "period", period)
