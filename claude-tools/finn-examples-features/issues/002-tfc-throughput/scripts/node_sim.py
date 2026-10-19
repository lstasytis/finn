"""Stitch a single node (or the full graph with deep FIFOs) from an existing build and
measure its rtlsim interval. Usage: node_sim.py <model.onnx> <node name | DEEPFIFO:<depth>> <N>"""
import copy
import sys

from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim

PART = "xczu3eg-sfvc784-2-e"
m = ModelWrapper(sys.argv[1])
what = sys.argv[2]
N = int(sys.argv[3])
if what.startswith("DEEPFIFO:"):
    depth = int(what.split(":")[1])
    for n in m.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            inst = getCustomOp(n)
            inst.set_nodeattr("depth", depth)
            inst.set_nodeattr("code_gen_dir_ipgen", "")
            inst.set_nodeattr("ipgen_path", "")
            inst.set_nodeattr("ip_path", "")
    m = m.transform(PrepareIP(PART, 10.0))
    # PrepareIP only regenerates FIFOs (others keep their code_gen dir); hls ipgen exists
    for n in m.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            inst = getCustomOp(n)
            inst.set_nodeattr("ipgen_path", inst.get_nodeattr("code_gen_dir_ipgen"))
            inst.set_nodeattr("ip_path", inst.get_nodeattr("code_gen_dir_ipgen"))
else:
    node = copy.deepcopy(m.get_node_from_name(what))
    vi_in = m.get_tensor_valueinfo(node.input[0])
    vi_out = m.get_tensor_valueinfo(node.output[0])
    inits = [x for x in m.graph.initializer if x.name in node.input[1:]]
    g = helper.make_graph([node], "single", [vi_in], [vi_out], initializer=inits)
    m2 = ModelWrapper(helper.make_model(g, producer_name="i002"))
    for t in list(node.input) + list(node.output):
        if t:
            m2.set_tensor_datatype(t, m.get_tensor_datatype(t))
    m = m2
import os
for ov in filter(None, os.environ.get("IPGEN_OVERRIDE", "").split(",")):
    nname, cdir = ov.split("=")
    inst = getCustomOp(m.get_node_from_name(nname))
    proj = cdir + "/project_" + nname
    inst.set_nodeattr("code_gen_dir_ipgen", cdir)
    inst.set_nodeattr("ipgen_path", proj)
    inst.set_nodeattr("ip_path", proj + "/sol1/impl/ip")
for p in list(m.model.metadata_props):
    if p.key in ("rtlsim_so", "vivado_stitch_proj", "wrapper_filename", "rtlsim_trace"):
        m.model.metadata_props.remove(p)
m = m.transform(CreateStitchedIP(PART, 10.0))
d = xsi_fifosim(m, N, max_iters=20000, behav=False)
ss = d["steady_state_cycles"] / max(1, d["steady_state_frames"])
print("RESULT", what, "N", N, "latency", d["latency_cycles"], "last_interval",
      d["interval_cycles"], "avg_interval", ss, "in_txns", d["N_IN_TXNS"], "out_txns",
      d["N_OUT_TXNS"], "timeout", d["TIMEOUT"], flush=True)
