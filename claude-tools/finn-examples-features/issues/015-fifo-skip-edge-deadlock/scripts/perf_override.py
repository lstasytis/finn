"""Issue 015 (4a): stitched-IP rtlsim throughput of a build with changed FIFO depths.
Takes the build's step_hw_ipgen.onnx, sets FIFO depths (SCALE=k multiplies all, MIN=n raises
all to n, NAME=DEPTH sets one), regenerates those FIFOs, stitches and runs xsi_fifosim
(behav=False, like step_measure_rtlsim_performance) for N frames.
MON=1: depth monitors on every FIFO and behavioral sim; prints each FIFO's peak occupancy
against its sized depth (the peaks are bounded by the new depths).
usage: perf_override.py <build dir> [N=3] [SCALE=k] [MIN=n] [MON=1] [PART=part] [NAME=DEPTH ...]"""
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim

part, clk = "xczu7ev-ffvc1156-2-e", 10.0
bdir = sys.argv[1]
kw = dict(a.split("=") for a in sys.argv[2:])
n = int(kw.pop("N", 3))
scale = float(kw.pop("SCALE", 1))
mind = int(kw.pop("MIN", 0))
part = kw.pop("PART", part)
mon = kw.pop("MON", "0") == "1"
model = ModelWrapper(bdir + "/intermediate_models/step_hw_ipgen.onnx")
changed = 0
sized = {}
for node in model.graph.node:
    if not node.op_type.startswith("StreamingFIFO"):
        continue
    inst = getCustomOp(node)
    d = inst.get_nodeattr("depth")
    new = max(int(d * scale), mind, int(kw.get(node.name, 0)))
    sized[node.name] = d
    if mon:
        inst.set_nodeattr("depth_monitor", 1)
    if new != d or mon:
        inst.set_nodeattr("depth", new)
        for a in ["code_gen_dir_ipgen", "ipgen_path", "ip_path"]:
            inst.set_nodeattr(a, "")
        changed += 1
print("FIFOs changed", changed, flush=True)
model = model.transform(PrepareIP(part, clk)).transform(HLSSynthIP())
model = model.transform(CreateStitchedIP(part, clk))
res = xsi_fifosim(model, n, behav=mon)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("maxcount")}), flush=True)
if mon:
    fifos = model.get_nodes_by_op_type("StreamingFIFO_rtl")
    for ind, node in enumerate(fifos):
        peak = res["maxcount" if ind == 0 else "maxcount_%d" % ind]
        prod, cons = model.find_producer(node.input[0]), model.find_consumer(node.output[0])
        flag = "  <-- above sized" if peak > sized[node.name] else ""
        print(f"{node.name:24s} {prod.name if prod else '-':34s} -> "
              f"{cons.name if cons else '-':34s} sized {sized[node.name]:6d} peak {peak:6d}{flag}")
model.save(os.environ["FINN_BUILD_DIR"] + "/perf_model_%d.onnx" % os.getpid())
