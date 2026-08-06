# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""rtlsim throughput of a sized graph, at whatever FIFO depths it carries.

The question the FIFO totals cannot answer: does the sized graph still run at
rate. Takes a ``step_set_fifo_depths.onnx`` checkpoint, optionally overwrites
its depths from another checkpoint's report, then runs ipgen, stitched IP and
the rtlsim performance measurement on it.

    python claude-tools/swg/swg_throughput.py --model cnv-w2a2 --tag baseline \
           --depths claude-tools/swg/goldens/cnv-w2a2_base.json
    python claude-tools/swg/swg_throughput.py --model cnv-w2a2 --tag candidate
"""

import argparse
import json
import os
import shutil
import sys

os.environ.setdefault(
    "FINN_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
sys.path.insert(0, os.path.dirname(__file__))

from qonnx.core.modelwrapper import ModelWrapper  # noqa: E402
from qonnx.custom_op.registry import getCustomOp  # noqa: E402
from swg_model_sizes import MODELS, build_dir  # noqa: E402

import finn.builder.build_dataflow as build  # noqa: E402
import finn.builder.build_dataflow_config as build_cfg  # noqa: E402


def apply_depths(src_onnx, report, dst_onnx):
    """Overwrite every StreamingFIFO depth from a ``fifo_report`` JSON."""
    model = ModelWrapper(src_onnx)
    depths = {k: v["depth"] for k, v in json.load(open(report))["fifos"].items()}
    n = 0
    for node in model.graph.node:
        if node.op_type.startswith("StreamingFIFO") and node.name in depths:
            getCustomOp(node).set_nodeattr("depth", depths[node.name])
            n += 1
    model.save(dst_onnx)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnv-w2a2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--depths", default=None, help="fifo_report JSON to force depths from")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    spec = MODELS[a.model]
    sized = os.path.join(build_dir(a.model), "intermediate_models", "step_set_fifo_depths.onnx")
    out = build_dir(a.model) + "_tp_" + a.tag
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out + "/intermediate_models", exist_ok=True)
    start = out + "/start.onnx"
    if a.depths:
        print("forced", apply_depths(sized, a.depths, start), "FIFO depths from", a.depths)
    else:
        shutil.copy(sized, start)

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=out,
        synth_clk_period_ns=spec.get("clk_ns", 5.0),
        board=spec["board"],
        shell_flow_type=(
            build_cfg.ShellFlowType.VITIS_ALVEO
            if spec["board"] == "U250"
            else build_cfg.ShellFlowType.VIVADO_ZYNQ
        ),
        rtlsim_batch_size=a.batch,
        steps=[
            "step_hw_codegen",
            "step_hw_ipgen",
            "step_create_stitched_ip",
            "step_measure_rtlsim_performance",
        ],
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
        ],
        verbose=True,
        enable_build_pdb_debug=False,
    )
    build.build_dataflow_cfg(start, cfg)
    rep = os.path.join(out, "report", "rtlsim_performance.json")
    print("REPORT", rep)
    if os.path.exists(rep):
        print(json.dumps(json.load(open(rep)), indent=2))


if __name__ == "__main__":
    main()
