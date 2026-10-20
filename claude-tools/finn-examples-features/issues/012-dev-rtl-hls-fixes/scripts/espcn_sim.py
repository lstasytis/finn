# Issue 012: copy of issue 007 subsim.py (only the ctx path changed). The MVAU_rtl sources come from
# $FINN_ROOT/finn-rtllib at stitch time, so running it from a worktree with the fixed RTL re-simulates
# the failing build's design with the fix.
# Stitched-IP rtlsim of a sub-graph of the espcn FIFO build (diagnostic for issue 007).
# usage: subsim.py <first node> <last node> <out dir> [fifo=depth,...]
# Input/reference come from the python execution of the generic HW model (ctx_hw.npz).
import json
import numpy as np
import os
import sys
import time
from copy import deepcopy
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

# optional input throttling "n,k": after n input transactions, pause k cycles
if os.environ.get("SUBSIM_THROTTLE"):
    from finn_xsi.sim_engine import SimEngine

    _thr = tuple(int(v) for v in os.environ["SUBSIM_THROTTLE"].split(","))
    _si = SimEngine.stream_input
    SimEngine.stream_input = lambda self, i, v, throttle=None: _si(self, i, v, _thr)
# optional output throttling "n,k": accept n outputs, then keep tready low k cycles
if os.environ.get("SUBSIM_OTHROTTLE"):
    from finn_xsi.sim_engine import SimEngine

    _othr = tuple(int(v) for v in os.environ["SUBSIM_OTHROTTLE"].split(","))

    def _collect(self, ostream, size, watchdog=None):
        class OutputCollector:
            def __init__(self, top):
                self.vld = top.get_bus_port(ostream, "tvalid")
                self.rdy = top.get_bus_port(ostream, "tready")
                self.dat = top.get_bus_port(ostream, "tdata")
                self.buf = []
                self.n = 0
                self.wait_until = 0

            def __iter__(self):
                return iter(self.buf)

            def __call__(self, sim):
                if self.rdy.as_bool():
                    if self.vld.read().as_bool():
                        if watchdog is not None:
                            watchdog.reset()
                        self.buf.append(self.dat.read().as_hexstr())
                        self.n += 1
                        if len(self.buf) == size:
                            return {self.rdy: "0"}
                        if self.n == _othr[0]:
                            self.n = 0
                            self.wait_until = sim.ticks + _othr[1]
                            return {self.rdy: "0"}
                    return {}
                if len(self.buf) < size:
                    if sim.ticks >= self.wait_until:
                        return {self.rdy: "1"}
                    return {}
                return None

        ret = OutputCollector(self)
        self.enlist(ret)
        return ret

    SimEngine.collect_output = _collect
S = "/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/s"
B = "/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_espcn_KV260_SOM_aiguk3l6"
E = "/home/lstasytis/finn/build/finn_build/fx/fifo/est_espcn_KV260_SOM_uu6ftob8"
SRC = os.environ.get("SUBSIM_MODEL", B + "/intermediate_models/step_create_stitched_ip.onnx")
PART = "xck26-sfvc784-2LV-c"
first, last, odir = sys.argv[1:4]
overrides = {}
if len(sys.argv) > 4 and sys.argv[4]:
    for kv in sys.argv[4].split(","):
        k, v = kv.split("=")
        overrides[k] = int(v)
os.makedirs(odir, exist_ok=True)

m = ModelWrapper(SRC)
names = [n.name for n in m.graph.node]
i0, i1 = names.index(first), names.index(last)

# reference tensors: generic HW model, compute nodes in order (Transpose lives in the parent)
g = ModelWrapper(E + "/intermediate_models/step_convert_to_hw.onnx")
ctx = np.load(S + "/ctx_hw.npz")
gnodes = [n for n in g.graph.node if n.op_type != "Transpose"]
skip = ("StreamingFIFO", "StreamingDataWidthConverter")
compute = [n.name for n in m.graph.node if not n.op_type.startswith(skip)]


def ref_out(spec_name):
    # output of a specialized compute node, from the generic model's context
    k = compute.index(spec_name)
    if spec_name.startswith("InnerShuffle"):
        return ctx[gnodes[k - 1].output[0]].transpose(0, 1, 3, 2)
    if spec_name.startswith("OuterShuffle"):
        k -= 1
    return ctx[gnodes[k].output[0]]


def last_compute_before(i):
    while m.graph.node[i].op_type.startswith(skip):
        i -= 1
    return m.graph.node[i].name


if i0 == 0:
    x = ctx[g.graph.node[0].output[0]]
else:
    x = ref_out(last_compute_before(i0 - 1))
ref = ref_out(last_compute_before(i1))

sub = deepcopy(m)
nodes = list(sub.graph.node)
for n in nodes[:i0] + nodes[i1 + 1 :]:
    sub.graph.node.remove(n)
iname, oname = nodes[i0].input[0], nodes[i1].output[0]
# weights/threshold initializers stay; replace the graph I/O
while len(sub.graph.input):
    sub.graph.input.pop()
while len(sub.graph.output):
    sub.graph.output.pop()


def vinfo(name):
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        if vi.name == name:
            return deepcopy(vi)


for vi in list(sub.graph.value_info):
    if vi.name in (iname, oname):
        sub.graph.value_info.remove(vi)
sub.graph.input.append(vinfo(iname))
sub.graph.output.append(vinfo(oname))
for p in list(sub.model.metadata_props):
    sub.model.metadata_props.remove(p)

# FIFO depth overrides: regenerate those FIFOs
# SUBSIM_TAPS="FIFO,..." regenerates those FIFOs with the diagnostic tap (logs to <odir>/<name>.tap)
taps = [t for t in os.environ.get("SUBSIM_TAPS", "").split(",") if t]
regen = []
for n in sub.graph.node:
    if n.name in overrides or n.name in taps:
        inst = getCustomOp(n)
        if n.name in overrides:
            inst.set_nodeattr("depth", overrides[n.name])
        if n.name in taps:
            inst.set_nodeattr("debug_log_path", os.path.abspath(os.path.join(odir, n.name + ".tap")))
        inst.set_nodeattr("code_gen_dir_ipgen", "")
        inst.set_nodeattr("ipgen_path", "")
        inst.set_nodeattr("ip_path", "")
        regen.append(n.name)
# behavioral MVU compute cores (faster rtlsim), everything else unchanged
if os.environ.get("SUBSIM_BEHAV_MVU"):
    import shutil

    _bl = os.environ["SUBSIM_BEHAV_MVU"].split(",")  # "1" = all, else node names
    for n in sub.get_nodes_by_op_type("MVAU_rtl"):
        if _bl != ["1"] and n.name not in _bl:
            continue
        inst = getCustomOp(n)
        src = inst.get_nodeattr("code_gen_dir_ipgen")
        dst = os.path.join(odir, "behav_" + n.name)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        wf = os.path.join(dst, n.name + "_wrapper.v")
        txt = open(wf).read()
        txt = txt.replace("`ifdef FINN_SIMULATION\n\t.FORCE_BEHAVIORAL(1),\n`endif\n", "\t.FORCE_BEHAVIORAL(1),\n")
        assert "FINN_SIMULATION" not in txt
        open(wf, "w").write(txt)
        for a in ("code_gen_dir_ipgen", "ipgen_path", "ip_path"):
            if inst.get_nodeattr(a) == src:
                inst.set_nodeattr(a, dst)
        print("behavioral", n.name, dst)
if regen:
    sub = sub.transform(PrepareIP(PART, 5.0))
    sub = sub.transform(HLSSynthIP())
print("sub-graph", names[i0], "..", names[i1], "nodes", len(sub.graph.node), "regen", regen)
x = x.reshape(sub.get_tensor_shape(iname)).astype(np.float32)
t = time.time()
sub = sub.transform(CreateStitchedIP(PART, 5.0))
print("stitch %.0fs" % (time.time() - t))
sub.set_metadata_prop("exec_mode", "rtlsim")
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles

perf = sub.transform(AnnotateCycles()).analysis(dataflow_performance)
sub.set_metadata_prop("rtlsim_liveness_estimate", str(int(perf["critical_path_cycles"] * 1.1 + 50)))
sub.save(odir + "/sub.onnx")
t = time.time()
out = oxe.execute_onnx(sub, {iname: x})[oname]
print("rtlsim %.0fs cycles %s" % (time.time() - t, sub.get_metadata_prop("cycles_rtlsim")))
ref = ref.reshape(out.shape)
bad = out != ref
np.save(odir + "/out.npy", out)
np.save(odir + "/ref.npy", ref)
res = dict(
    first=first,
    last=last,
    overrides=overrides,
    frac_bad=float(bad.mean()),
    n_bad=int(bad.sum()),
    zeros_out=int((out == 0).sum()),
    zeros_ref=int((ref == 0).sum()),
    cycles=sub.get_metadata_prop("cycles_rtlsim"),
)
print(json.dumps(res))
json.dump(res, open(odir + "/result.json", "w"))
