"""Fast loop for issue 005: run maximize SetFolding on a convert checkpoint, log each probe,
and score the committed folding with the optimizer's resource view."""
import argparse, glob, json, sys, time, warnings
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import finn.transformation.fpgadataflow.set_folding as sf
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("board")
ap.add_argument("--target", type=int, default=None)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--wwidth", type=int, default=1024)
ap.add_argument("--root", default="/home/lstasytis/finn/build/finn_build/fx/folding")
ap.add_argument("--prefix", default="bnn-pynq_")
ap.add_argument("--committed-only", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--save", default=None)
a = ap.parse_args()
est = sorted(glob.glob(f"{a.root}/est_{a.prefix}{a.model}_{a.board}_*/intermediate_models"))
est = [e for e in est if not e.endswith("_lock")][0].rsplit("/", 1)[0]
print("est dir", est)

def view(m, folder):
    m = m.transform(InsertDWC()).transform(SpecializeLayers(folder.fpgapart))
    m = m.transform(AnnotateCycles())
    res = folder._aggregate_resources(m)
    cyc = m.analysis(dataflow_performance)["max_cycles"]
    return cyc, {k: res.get(k, 0) for k in folder.target_resources}, m

folder = sf.SetFolding(a.target, style="optimizer", platform=a.board, seed=a.seed, folding_effort=50,
                       mvau_wwidth_max=a.wwidth, auto_fifo_strategy="heuristic_analytical")
folder._budgets = folder._resource_budgets()
print("budgets", folder._budgets)
committed = ModelWrapper(est + "/intermediate_models/step_generate_estimate_reports.onnx")
cyc, res, cm = view(committed, folder)
print("COMMITTED view: cycles", cyc, "res", res, "fits", folder._fits(res, folder._budgets),
      "cost", round(folder._cost(res, folder._budgets), 3))
for n in cm.graph.node:
    if n.op_type.startswith(("MVAU", "VVAU")):
        i = getCustomOp(n)
        print("   ", n.name, "PE", i.get_nodeattr("PE"), "SIMD", i.get_nodeattr("SIMD"), i.get_nodeattr("mem_mode"),
              i.get_nodeattr("ram_style"), "wmem", i.calc_wmem(), i.node_res_estimation(folder.fpgapart))
if a.committed_only:
    sys.exit()

orig = sf.SetFolding._fold_at
log = []
def fold_at(self, opt_template, targets, target_cycles, padding, seed=None):
    t0 = time.time()
    r = orig(self, opt_template, targets, target_cycles, padding, seed)
    rec = dict(target=target_cycles, pad=padding, achieved=r.achieved, fits=r.fits,
               res={k: round(r.metrics.get(k, 0), 1) for k in self.target_resources}, cost=round(r.cost, 3),
               s=round(time.time() - t0, 1))
    log.append(rec); print("PROBE", rec, flush=True)
    return r
sf.SetFolding._fold_at = fold_at
model = ModelWrapper(est + "/intermediate_models/phase_convert_to_hardware.onnx")
t0 = time.time()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = model.transform(folder, apply_to_subgraphs=True)
for x in w:
    if "SetFolding" in str(x.message):
        print("WARN", x.message)
print("time", round(time.time() - t0, 1))
print("REPORT", json.dumps(folder.last_search_report, default=str))
if a.save:
    model.save(a.save)
cyc, res, fm = view(model, folder)
print("RESULT view: cycles", cyc, "res", res)
for n in fm.graph.node:
    if n.op_type.startswith(("MVAU", "VVAU")):
        i = getCustomOp(n)
        print("   ", n.name, "PE", i.get_nodeattr("PE"), "SIMD", i.get_nodeattr("SIMD"), i.get_nodeattr("mem_mode"),
              i.get_nodeattr("ram_style"), "wmem", i.calc_wmem(), "cyc", i.get_exp_cycles(), i.node_res_estimation(folder.fpgapart))
if a.out:
    json.dump(dict(probes=log, report=folder.last_search_report, cycles=cyc, res=res), open(a.out, "w"), default=str, indent=1)
