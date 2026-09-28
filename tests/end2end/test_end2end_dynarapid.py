# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Bitfile build with DynaRapid place-and-route (dynarapid_pnr=True) of a small MLP on the
ZCU104: builder integration, reports, and a functional check of the DynaRapid-routed
accelerator (post-route netlist vs FINN's stitched-IP RTL, experiments/dynarapid)."""

import pytest

import json
import os
import subprocess
import sys
import torch
import torch.nn as nn
from brevitas.export import export_qonnx
from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.test import load_test_checkpoint_or_skip

BOARD = "ZCU104"
CLK_NS = 5.0
build_dir = os.environ["FINN_BUILD_DIR"]
export_onnx = build_dir + "/end2end_dynarapid_mlp_export.onnx"
build_out = build_dir + "/end2end_dynarapid_mlp_build"


def dynarapid_available():
    root = os.environ.get("DYNARAPID_ROOT", "")
    return os.path.isdir(os.path.join(root, "build", "classes", "java", "main"))


pytestmark = [
    pytest.mark.end2end,
    pytest.mark.dynarapid,
    pytest.mark.xdist_group(name="end2end_dynarapid"),
]


def test_end2end_dynarapid_export():
    torch.manual_seed(0)
    net = nn.Sequential(
        QuantLinear(64, 32, bias=True, weight_bit_width=2),
        nn.BatchNorm1d(32),
        QuantReLU(bit_width=2),
        QuantLinear(32, 32, bias=True, weight_bit_width=2),
        nn.BatchNorm1d(32),
        QuantReLU(bit_width=2),
        QuantLinear(32, 8, bias=True, weight_bit_width=2),
        QuantIdentity(bit_width=4),
    )
    net.eval()
    # integer input straight into the first layer: no float pre-processing layers, whose
    # Xilinx floating-point cores (encrypted IP) DynaRapid does not support
    export_qonnx(net, torch.randint(-128, 128, (1, 64)).float(), export_path=export_onnx)
    model = ModelWrapper(export_onnx)
    model.set_tensor_datatype(model.get_first_global_in(), DataType["INT8"])
    model.save(export_onnx)
    qonnx_cleanup(export_onnx, out_file=export_onnx)
    # top-1 class (LabelSelect): the output quantizer's scale is absorbed into it instead
    # of remaining as a float multiplication
    model = ModelWrapper(export_onnx)
    model = model.transform(InsertTopK(k=1))
    model.save(export_onnx)
    assert os.path.isfile(export_onnx)


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_dynarapid_build():
    if not dynarapid_available():
        pytest.skip("DynaRapid is not built (DYNARAPID_ROOT)")
    load_test_checkpoint_or_skip(export_onnx)
    cfg = build.DataflowBuildConfig(
        output_dir=build_out,
        target_fps=100000,
        synth_clk_period_ns=CLK_NS,
        board=BOARD,
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        dynarapid_pnr=True,
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )
    build.build_dataflow_cfg(export_onnx, cfg)
    for f in (
        "bitfile/finn-accel.bit",
        "bitfile/finn-accel.hwh",
        "report/post_synth_resources.json",
        "report/post_route_timing.rpt",
        "driver/driver.py",
        "dynarapid_pnr/dynarapid_zynq.json",
    ):
        assert os.path.isfile(os.path.join(build_out, f)), f
    res = json.load(open(os.path.join(build_out, "dynarapid_pnr", "dynarapid_zynq.json")))
    assert res["status"] == "ok"
    assert res["nets_with_routing_errors"] == 0
    assert res["wns_ns"] is not None and res["wns_ns"] >= 0
    # per-layer resources are found in the assembled design under the FINN node names
    post_synth = json.load(open(os.path.join(build_out, "report", "post_synth_resources.json")))
    assert "(top)" in post_synth
    assert any(k.startswith("StreamingDataflowPartition_1_MVAU") for k in post_synth)
    # the hardware handoff describes the IODMAs the driver programs
    hwh = open(os.path.join(build_out, "bitfile", "finn-accel.hwh")).read()
    assert "idma0" in hwh and "odma0" in hwh


@pytest.mark.slow
@pytest.mark.vivado
def test_end2end_dynarapid_functional():
    """Post-route netlist of the DynaRapid accelerator (with the shell's AXI bridges) and
    FINN's stitched-IP RTL of the same graph produce identical output buffers."""
    accel_dir = os.path.join(build_out, "dynarapid_pnr")
    load_test_checkpoint_or_skip(os.path.join(accel_dir, "accel.onnx"))
    script = os.path.join(
        os.environ["FINN_ROOT"], "experiments", "dynarapid", "verify_accel.py"
    )
    out = os.path.join(build_out, "verify_accel")
    subprocess.run(
        [sys.executable, script, "--accel-dir", accel_dir, "--out", out, "--frames", "4"],
        check=True,
    )
    res = json.load(open(os.path.join(out, "verify_accel.json")))
    assert res["dr_done"] and res["ref_done"]
    assert res["ref_wrote_output"]
    assert res["outputs_match"]
