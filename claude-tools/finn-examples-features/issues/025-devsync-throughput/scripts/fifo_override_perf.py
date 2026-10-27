"""Re-stitch a built FINN model (optionally with every StreamingFIFO depth overridden)
and measure stitched-IP rtlsim throughput (behav=False, same as
step_measure_rtlsim_performance) for several batch sizes.

usage: python fifo_override_perf.py <model.onnx> <tag> <fifo_depth|0> <N,N,...>
fifo_depth 0 = keep the depths of the build.
"""
import json
import os
import sys
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.util.rtlsim import annotate_rtlsim_performance

PART = "xczu3eg-sfvc784-2-e"
CLK = 10.0

src, tag, depth, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
out_dir = os.path.join(os.environ["FINN_BUILD_DIR"], "results")
os.makedirs(out_dir, exist_ok=True)
m = ModelWrapper(src)
for n in m.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        inst = getCustomOp(n)
        if depth > 0:
            inst.set_nodeattr("depth", depth)
            # force regeneration in our own build dir
            inst.set_nodeattr("code_gen_dir_ipgen", "")
            inst.set_nodeattr("ipgen_path", "")
            inst.set_nodeattr("ip_path", "")
        print(n.name, inst.get_nodeattr("depth"), inst.get_nodeattr("ram_style"))
if depth > 0:
    m = m.transform(PrepareIP(PART, CLK))
    m = m.transform(HLSSynthIP())
m = m.transform(CreateStitchedIP(PART, CLK))
m.save(os.path.join(out_dir, tag + ".onnx"))
res = {}
for N in [int(x) for x in ns.split(",")]:
    d = xsi_fifosim(m, N, max_iters=100000, behav=False)
    d = {k: v for k, v in d.items() if "max_count" not in k}
    d = annotate_rtlsim_performance(d, N, CLK)
    avg = d["steady_state_cycles"] / max(1, d["steady_state_frames"])
    d["avg_steady_interval"] = avg
    print("RESULT", tag, "N=%d" % N, "interval(max)=", d["interval_cycles"],
          "avg_steady=", avg, "latency=", d["latency_cycles"], "cycles=", d["cycles"])
    res[N] = d
with open(os.path.join(out_dir, tag + ".json"), "w") as f:
    json.dump(res, f, indent=2)
