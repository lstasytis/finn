"""Re-run step_set_fifo_depths (heuristic_analytical) on a build and log every edge_depth call.
size_dbg.py <build dir> [out dir]"""
import inspect
import json
import os
import pickle
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.builder.build_dataflow_config as bc
import finn.transformation.fpgadataflow.derive_characteristic as dc
from finn.builder.build_dataflow_steps import step_set_fifo_depths

PARTS = {"U55C": "xcu55c-fsvh2892-2L-e", "ZCU104": "xczu7ev-ffvc1156-2-e", "AUP-ZU3_8GB": "xczu3eg-sfvc784-2-e", "KV260_SOM": "xck26-sfvc784-2LV-c"}
bdir = sys.argv[1]
out = sys.argv[2]
board = [b for b in PARTS if b in bdir][0]
orig = dc.edge_depth
log = []


def logged(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor=0, refill=0):
    f = inspect.currentframe().f_back
    tensor = f.f_locals.get("tensor")
    causal = f.f_back.f_locals.get("causal")
    d = orig(peak, read_times, edge, fill_only, throttleable, gp, pp, idle_floor, refill)
    allowance = edge.per_frame * (1.0 - min(edge.t_up, gp) / float(gp))
    floor = min(peak, dc.burst_above_rate(read_times, edge.per_frame / float(gp)))
    log.append(dict(w=edge.writes, r=read_times, cons=f.f_locals.get('consumer').name if f.f_locals.get('consumer') is not None else None, t=tensor, causal=causal, peak=int(peak), allow=round(allowance, 1),
                    floor=int(floor), thr=throttleable, t_up=int(edge.t_up),
                    pf=int(edge.per_frame), depth=int(d), gp=gp, refill=int(refill), t_full=int(edge.t_full), slack=int(f.f_locals.get("slack", -1))))
    return d


dc.edge_depth = logged
os.makedirs(out, exist_ok=True)
cfg = bc.DataflowBuildConfig(output_dir=out, synth_clk_period_ns=10.0, fpga_part=PARTS[board],
                             generate_outputs=[], auto_fifo_depths=True,
                             auto_fifo_strategy="heuristic_analytical")
m = ModelWrapper(bdir + "/intermediate_models/step_transpose_decomposition.onnx")
m = step_set_fifo_depths(m, cfg)
pickle.dump(log, open(out + '/edges.pkl', 'wb'))
for e in log:
    if e["t"] and (e["cons"] or "").startswith("ElementwiseAdd"):
        print("RESIDUAL", json.dumps(dict(t=e["t"], cons=e["cons"], causal=e["causal"], w0=int(min(e["w"])), w1=int(max(e["w"])), r0=int(min(e["r"])), r1=int(max(e["r"])), nw=len(e["w"]), nr=len(e["r"]))))
    print('EDGE', json.dumps({k: v for k, v in e.items() if k not in ('w', 'r')}))
for n in m.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        p = m.find_producer(n.input[0])
        c = m.find_consumer(n.output[0])
        print("FIFO", n.name, getCustomOp(n).get_nodeattr("depth"), p.name if p else "-", "->",
              c.name if c else "-", n.input[0])
