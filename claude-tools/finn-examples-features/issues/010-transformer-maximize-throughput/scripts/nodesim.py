"""Steady-state cycles per frame of a node range of a finished build, by stitched-IP rtlsim
with dummy data (several frames), vs the nodes' get_exp_cycles().
nodesim.py <build dir> <first node> <last node> [N_FRAMES] [first..last are graph-order indices
or node names]. Reuses the build's generated IP (read only); the stitched project goes to
$FINN_BUILD_DIR. Dangling inputs/outputs of the range become graph I/O (dynamic MVAU weights,
multi-output Split/Dup, multi-input Concat/Add).  Prints one RESULT line (JSON)."""
import json
import os
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.util.basic import pynq_part_map

B, first, last = sys.argv[1:4]
nfr = int(sys.argv[4]) if len(sys.argv) > 4 else 4
part = pynq_part_map[os.environ.get("NODESIM_BOARD", "ZCU104")]
clk = float(os.environ.get("NODESIM_CLK", "10.0"))
m = ModelWrapper(B + "/intermediate_models/step_create_stitched_ip.onnx")
names = [n.name for n in m.graph.node]
i0 = int(first) if first.isdigit() else names.index(first)
i1 = int(last) if last.isdigit() else names.index(last)
sub = deepcopy(m)
nodes = list(sub.graph.node)
keep = nodes[i0 : i1 + 1]
for n in nodes[:i0] + nodes[i1 + 1 :]:
    sub.graph.node.remove(n)
inits = {t.name for t in sub.graph.initializer}
produced = {o for n in keep for o in n.output}
consumed = {i for n in keep for i in n.input}
ins = [i for n in keep for i in n.input if i not in produced and i not in inits and i != ""]
outs = [o for n in keep for o in n.output if o not in consumed]
# unused initializers (and their value_info) go
for t in list(sub.graph.initializer):
    if t.name not in consumed:
        sub.graph.initializer.remove(t)


def vinfo(name):
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.name == name:
            return deepcopy(vi)
    raise KeyError(name)


while len(sub.graph.input):
    sub.graph.input.pop()
while len(sub.graph.output):
    sub.graph.output.pop()
for vi in list(sub.graph.value_info):
    if vi.name in ins + outs:
        sub.graph.value_info.remove(vi)
for i in ins:
    sub.graph.input.append(vinfo(i))
for o in outs:
    sub.graph.output.append(vinfo(o))
for p in list(sub.model.metadata_props):
    sub.model.metadata_props.remove(p)
exp = {}
for n in keep:
    if n.op_type.startswith("StreamingFIFO"):
        continue
    exp[n.name] = int(getCustomOp(n).get_exp_cycles())
t0 = time.time()
sub = sub.transform(CreateStitchedIP(part, clk))
t1 = time.time()
mx = max(exp.values())
d = xsi_fifosim(sub, nfr, max_iters=max(20000, 4 * mx * nfr), behav=False)
t2 = time.time()
ss = d["steady_state_cycles"] / max(1, d["steady_state_frames"])
res = dict(
    first=names[i0],
    last=names[i1],
    exp=exp,
    max_exp=mx,
    rtlsim=ss,
    ratio=round(ss / mx, 4),
    latency=d["latency_cycles"],
    frames=nfr,
    ss_frames=d["steady_state_frames"],
    timeout=d["TIMEOUT"],
    stitch_s=round(t1 - t0),
    sim_s=round(t2 - t1),
    proj=sub.get_metadata_prop("vivado_stitch_proj"),
)
print("RESULT", json.dumps(res), flush=True)
if os.environ.get("NODESIM_KEEP") != "1":
    # keep the disk small: the stitched project and the compiled sim are scratch
    shutil.rmtree(sub.get_metadata_prop("vivado_stitch_proj"), ignore_errors=True)
    so = sub.get_metadata_prop("rtlsim_so")
    if so:
        shutil.rmtree(so.split("/xsim.dir")[0], ignore_errors=True)
