"""Fast loop (issue 021): maximize SetFolding on a reference build's convert checkpoint, as
step_target_fps_parallelization runs it in the fold-max test (effort 50, seed 1), and the
committed folding scored the same way.
    python probe_max.py <est build dir> <board> [--out result.json]"""
import argparse
import json
import time
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.transformation.fpgadataflow.set_folding as sf
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles

ap = argparse.ArgumentParser()
ap.add_argument("est")
ap.add_argument("board")
ap.add_argument("--out", default=None)
ap.add_argument("--lut", type=float, default=None, help="LUT limit instead of RESOURCE_LIMITS")
a = ap.parse_args()
if a.lut is not None:
    sf.SetFolding.RESOURCE_LIMITS = dict(sf.SetFolding.RESOURCE_LIMITS, LUT=a.lut)

folder = sf.SetFolding(None, style="optimizer", platform=a.board, seed=1, folding_effort=50,
                       auto_fifo_strategy="heuristic_analytical")
folder._budgets = folder._resource_budgets()
committed = ModelWrapper(a.est + "/intermediate_models/step_generate_estimate_reports.onnx")
if hasattr(sf, "weight_info"):
    folder._weights = sf.weight_info(committed)
ref = folder._resources_as_built(committed)
ref_cycles = committed.transform(AnnotateCycles()).analysis(dataflow_performance)["max_cycles"]
print("budget", {k: round(v) for k, v in folder._budgets.items()})
print("committed: cycles", ref_cycles, "as built", {k: round(ref.get(k, 0)) for k in folder.target_resources},
      "fits", folder._fits(ref, folder._budgets))
m = ModelWrapper(a.est + "/intermediate_models/phase_convert_to_hardware.onnx")
t0 = time.time()
m = m.transform(folder)
rep = folder.last_search_report
dt = time.time() - t0
print("maximize: cycles", rep["achieved_cycles_per_frame"], "x ref %.3f" % (ref_cycles / rep["achieved_cycles_per_frame"]),
      "fits", rep["fits_budget"], "as built", {k: round(v) for k, v in rep["resources_as_built"].items()},
      "%.0f s" % dt)
for n in m.graph.node:
    if n.op_type.startswith(("MVAU", "VVAU")):
        i = getCustomOp(n)
        print("   ", n.name, "PE", i.get_nodeattr("PE"), "SIMD", i.get_nodeattr("SIMD"),
              i.get_nodeattr("ram_style"), "wmem", i.calc_wmem())
if a.out:
    json.dump(dict(ref_cycles=ref_cycles, ref_as_built=ref, report=rep, seconds=dt), open(a.out, "w"),
              indent=1, default=str)
