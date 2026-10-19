"""Offline re-run of heuristic_analytical sizing on a build with rule toggles.
evalrule.py <build dir> -> one JSON line per config"""
import json
import os
import sys
import tempfile

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.builder.build_dataflow_config as bc
from finn.builder.build_dataflow_steps import step_set_fifo_depths

PARTS = {
    "AUP-ZU3_8GB": "xczu3eg-sfvc784-2-e",
    "KV260_SOM": "xck26-sfvc784-2LV-c",
    "ZCU104": "xczu7ev-ffvc1156-2-e",
    "U55C": "xcu55c-fsvh2892-2L-e",
    "V80": "xcv80-lsva4737-2MHP-e-S",
    "VCK190": "xcvc1902-vsva2197-2MP-e-S",
}
bdir = sys.argv[1]
board = [b for b in PARTS if b in bdir][0]
res = {}
for cfgname, rn, rd in [("base", "0", "0"), ("RN", "1", "0"), ("RD", "0", "1"), ("both", "1", "1")]:
    os.environ["FIX006_RN"], os.environ["FIX006_RD"] = rn, rd
    out = tempfile.mkdtemp(dir=os.environ["FINN_BUILD_DIR"], prefix="evalrule_")
    cfg = bc.DataflowBuildConfig(output_dir=out, synth_clk_period_ns=10.0, fpga_part=PARTS[board],
                                 generate_outputs=[], auto_fifo_depths=True,
                                 auto_fifo_strategy="heuristic_analytical")
    m = ModelWrapper(bdir + "/intermediate_models/step_transpose_decomposition.onnx")
    m = step_set_fifo_depths(m, cfg)
    fifos = {}
    bits = 0
    for n in m.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            i = getCustomOp(n)
            fifos[n.name] = i.get_nodeattr("depth")
            bits += i.get_nodeattr("depth") * i.get_instream_width()
    res[cfgname] = dict(kbit=bits / 1024, depths=fifos)
base = res["base"]["depths"]
for k, v in res.items():
    diff = {n: (base[n], d) for n, d in v["depths"].items() if base.get(n) != d}
    print("EVAL", json.dumps(dict(build=bdir.split("/fx/")[1], cfg=k, kbit=round(v["kbit"], 2), diff=diff)), flush=True)
