"""Issue 018 (after 015's perf_override.py): stitched-IP rtlsim throughput of a copied build
(prep_copy.py) with changed FIFO depths. Depths: DEPTHS=<json {fifo name: depth}>, NAME=DEPTH,
CAP=n (every FIFO deeper than n set to n), SET2=a,b,... (these FIFOs at 2). MON=1: depth
monitors, records each FIFO's peak occupancy (bounded by its depth; with BEHAV=1 the FIFOs are
the unbounded behavioral fifo_gauge, a large-FIFO diagnostic). Writes the result to OUT=<json>.
usage: perf.py <copy dir> OUT=<json> [N=2] [DEPTHS=..] [CAP=n] [MON=1] [NAME=DEPTH ...]"""
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim

part, clk = "xcu55c-fsvh2892-2L-e", 3.0  # PART=<part> overrides
cdir = sys.argv[1]
kw = dict(a.split("=", 1) for a in sys.argv[2:])
out = kw.pop("OUT")
n = int(kw.pop("N", 2))
part = kw.pop("PART", part)
cap = int(kw.pop("CAP", 0))
mon = kw.pop("MON", "0") == "1"
# BEHAV=1: behavioral sim, FIFOs unbounded (fifo_gauge); else the synthesizable fifo.sv
behav = kw.pop("BEHAV", "0") == "1"
want = json.load(open(kw.pop("DEPTHS"))) if "DEPTHS" in kw else {}
for name in [s for s in kw.pop("SET2", "").split(",") if s]:
    want[name] = 2
want.update({k: int(v) for k, v in kw.items()})
model = ModelWrapper(cdir + "/step_hw_ipgen.onnx")
sized, changed = {}, {}
for node in model.graph.node:
    if not node.op_type.startswith("StreamingFIFO"):
        continue
    inst = getCustomOp(node)
    d = inst.get_nodeattr("depth")
    new = int(want.get(node.name, d))
    if cap and new > cap:
        new = cap
    sized[node.name] = new
    if mon:
        inst.set_nodeattr("depth_monitor", 1)
    if new != d or mon:
        inst.set_nodeattr("depth", new)
        for a in ["code_gen_dir_ipgen", "ipgen_path", "ip_path"]:
            inst.set_nodeattr(a, "")
        if new != d:
            changed[node.name] = (d, new)
print("FIFOs changed", changed, flush=True)
model = model.transform(PrepareIP(part, clk)).transform(HLSSynthIP())
model = model.transform(CreateStitchedIP(part, clk))
# watchdog: 4x the estimated critical path (the builder uses 1.1x), stalls included
lat = model.transform(AnnotateCycles()).analysis(dataflow_performance)["critical_path_cycles"]
res = xsi_fifosim(model, n, max_iters=int(4 * lat * n), behav=behav)
row = dict(changed=changed, **{k: v for k, v in res.items() if not k.startswith("maxcount")})
if mon:
    peaks = {}
    for ind, node in enumerate(model.get_nodes_by_op_type("StreamingFIFO_rtl")):
        peaks[node.name] = res["maxcount" if ind == 0 else "maxcount_%d" % ind]
    row["peaks"] = peaks
row["bits"] = sum(
    sized[nd.name] * getCustomOp(nd).get_instream_width()
    for nd in model.graph.node
    if nd.op_type.startswith("StreamingFIFO")
)
json.dump(row, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in row.items() if k != "peaks"}), flush=True)
