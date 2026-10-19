"""Re-stitch an existing build with changed FIFO depths and measure the rtlsim interval.
fifo_exp.py <model.onnx> gauge[:throttle]|set:<FIFO=depth,...|ALL=depth> <N,...>
gauge: all FIFOs 8192 deep with depth monitors, prints max occupancy per FIFO."""
import os
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim

PART = os.environ.get("PART", "xczu3eg-sfvc784-2-e")
m = ModelWrapper(sys.argv[1])
mode = sys.argv[2]
NS = [int(x) for x in sys.argv[3].split(",")]
fifos = [n for n in m.graph.node if n.op_type.startswith("StreamingFIFO")]
orig = {n.name: getCustomOp(n).get_nodeattr("depth") for n in fifos}
gauge = mode.startswith("gauge")
new = {}
if gauge:
    new = {n.name: 8192 for n in fifos}
else:
    for kv in mode.split(":", 1)[1].split(","):
        k, v = kv.split("=")
        if k == "ALL":
            new.update({n.name: max(int(v), orig[n.name]) for n in fifos})
        else:
            new[k] = int(v)
for n in fifos:
    inst = getCustomOp(n)
    if n.name in new or gauge:
        inst.set_nodeattr("depth", new.get(n.name, orig[n.name]))
        inst.set_nodeattr("depth_monitor", 1 if gauge else 0)
        inst.set_nodeattr("code_gen_dir_ipgen", "")
        inst.set_nodeattr("ipgen_path", "")
        inst.set_nodeattr("ip_path", "")
regen = [x for x in os.environ.get("REGEN", "").split(",") if x]
for name in regen:
    inst = getCustomOp(m.get_node_from_name(name))
    for a in ("code_gen_dir_ipgen", "ipgen_path", "ip_path"):
        inst.set_nodeattr(a, "")
m = m.transform(PrepareIP(PART, 10.0))
if regen:
    from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
    m = m.transform(HLSSynthIP())
for n in fifos:
    inst = getCustomOp(n)
    if n.name in new or gauge:
        inst.set_nodeattr("ipgen_path", inst.get_nodeattr("code_gen_dir_ipgen"))
        inst.set_nodeattr("ip_path", inst.get_nodeattr("code_gen_dir_ipgen"))
for p in list(m.model.metadata_props):
    if p.key in ("rtlsim_so", "vivado_stitch_proj", "wrapper_filename", "rtlsim_trace"):
        m.model.metadata_props.remove(p)
m = m.transform(CreateStitchedIP(PART, 10.0))
throttle = 0
if mode == "gauge:throttle":
    max_cycles = m.analysis(dataflow_performance)["max_cycles"]
    first = getCustomOp(m.graph.node[0])
    throttle = max(0, max_cycles - int(np.prod(first.get_folded_input_shape()[:-1])))
for N in NS:
    d = xsi_fifosim(m, N, max_iters=int(os.environ.get("MAXIT", "300000")), throttle_cycles=throttle, behav=gauge)
    ss = d["steady_state_cycles"] / max(1, d["steady_state_frames"])
    print("RESULT", mode[:60], "N", N, "latency", d["latency_cycles"], "last_interval",
          d["interval_cycles"], "avg_interval", ss, "timeout", d["TIMEOUT"], flush=True)
    if gauge:
        for i, n in enumerate(fifos):
            key = "maxcount" if i == 0 else "maxcount_%d" % i
            print("GAUGE N", N, n.name, "orig", orig[n.name], "max", d.get(key), flush=True)
