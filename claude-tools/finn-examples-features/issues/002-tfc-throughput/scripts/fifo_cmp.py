"""Re-run step_set_fifo_depths (heuristic_analytical) on a build's step_hw_ipgen
checkpoint and print FIFO depths + estimate. BASELINE=1 undoes the fix002 overhead."""
import json
import os
import sys

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.builder.build_dataflow_config as bc
from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.builder.build_dataflow_steps import step_set_fifo_depths
from finn.custom_op.fpgadataflow.hls.labelselect_hls import LabelSelect_hls
from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.custom_op.fpgadataflow.labelselect import LabelSelect
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU

if os.environ.get("BASELINE") == "1":
    MVAU_hls.get_exp_cycles = MVAU.get_exp_cycles
    LabelSelect_hls.get_exp_cycles = LabelSelect.get_exp_cycles
PARTS = {
    "AUP-ZU3_8GB": "xczu3eg-sfvc784-2-e",
    "KV260_SOM": "xck26-sfvc784-2LV-c",
    "ZCU104": "xczu7ev-ffvc1156-2-e",
    "U55C": "xcu55c-fsvh2892-2L-e",
}
bdir, out = sys.argv[1], sys.argv[2]
board = [b for b in PARTS if b in bdir][0]
fclk = json.load(open(bdir + "/report/rtlsim_performance.json"))["fclk[mhz]"]
cfg = bc.DataflowBuildConfig(
    output_dir=out,
    synth_clk_period_ns=1000.0 / fclk,
    fpga_part=PARTS[board],
    generate_outputs=[],
    auto_fifo_depths=True,
    auto_fifo_strategy="heuristic_analytical",
)
m = ModelWrapper(bdir + "/intermediate_models/step_transpose_decomposition.onnx")
m = step_set_fifo_depths(m, cfg)
depths = [
    (n.name, getCustomOp(n).get_nodeattr("depth"))
    for n in m.graph.node
    if n.op_type.startswith("StreamingFIFO")
]
bits = sum(
    getCustomOp(n).get_nodeattr("depth") * getCustomOp(n).get_instream_width()
    for n in m.graph.node
    if n.op_type.startswith("StreamingFIFO")
)
perf = m.analysis(dataflow_performance)
print("RESULT", json.dumps(dict(max_cycles=perf["max_cycles"], kbit=bits / 1024, depths=depths)))
