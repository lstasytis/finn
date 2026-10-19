"""Print heuristic_analytical FIFO depths of a build (worktree code as is).
sizeonly.py <build dir>"""
import json
import os
import sys
import tempfile

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.builder.build_dataflow_config as bc
from finn.builder.build_dataflow_steps import step_set_fifo_depths

PARTS = {"AUP-ZU3_8GB": "xczu3eg-sfvc784-2-e", "KV260_SOM": "xck26-sfvc784-2LV-c",
         "ZCU104": "xczu7ev-ffvc1156-2-e", "U55C": "xcu55c-fsvh2892-2L-e"}
bdir = sys.argv[1]
board = [b for b in PARTS if b in bdir][0]
out = tempfile.mkdtemp(dir=os.environ["FINN_BUILD_DIR"], prefix="sizeonly_")
cfg = bc.DataflowBuildConfig(output_dir=out, synth_clk_period_ns=10.0, fpga_part=PARTS[board],
                             generate_outputs=[], auto_fifo_depths=True,
                             auto_fifo_strategy="heuristic_analytical")
m = step_set_fifo_depths(ModelWrapper(bdir + "/intermediate_models/step_transpose_decomposition.onnx"), cfg)
d, bits = {}, 0
for n in m.graph.node:
    if n.op_type.startswith("StreamingFIFO"):
        i = getCustomOp(n)
        d[n.name] = i.get_nodeattr("depth")
        bits += i.get_nodeattr("depth") * i.get_instream_width()
print("SIZE", json.dumps(dict(kbit=round(bits / 1024, 2), depths=d)))
