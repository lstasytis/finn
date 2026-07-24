# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

"""Tests for the AlignLabels layer and its FIFO side-channel sizing.

AlignLabels streams the model output (labels) together with the model input that
produced it: ``InsertAlignLabels`` forks the input with a DuplicateStreams and
re-joins it at the output with an AlignLabels node, so the accelerator emits two
streams -- the labels and the (delayed) inputs. The bypass FIFO between the fork
and the join has to buffer the model-latency worth of input while the model
computes (otherwise the fork back-pressures and stalls the model path), which the
analytic FIFO sizer must recognise and size (AlignLabels is a newly named join
node, so the branch heuristic has to be taught about it).
"""

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers

test_fpga_part = "xc7z020clg400-1"
target_clk_ns = 10.0


def make_alignlabels_modelwrapper(n_label, n_data, pe, label_dt, data_dt):
    lab = helper.make_tensor_value_info("lab", TensorProto.FLOAT, [1, n_label])
    dat = helper.make_tensor_value_info("dat", TensorProto.FLOAT, [1, n_data])
    olab = helper.make_tensor_value_info("olab", TensorProto.FLOAT, [1, n_label])
    odat = helper.make_tensor_value_info("odat", TensorProto.FLOAT, [1, n_data])
    node = helper.make_node(
        "AlignLabels",
        ["lab", "dat"],
        ["olab", "odat"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        label_dtype=label_dt.name,
        data_dtype=data_dt.name,
        label_shape=[1, n_label],
        data_shape=[1, n_data],
        PE=pe,
    )
    graph = helper.make_graph([node], "alignlabels_graph", [lab, dat], [olab, odat])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="alignlabels-model"))
    model.set_tensor_datatype("lab", label_dt)
    model.set_tensor_datatype("dat", data_dt)
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


# label / data element counts
@pytest.mark.parametrize("n_label, n_data", [(10, 64), (4, 48)])
# folding of the data stream
@pytest.mark.parametrize("pe", [1, 8])
# execution mode
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_alignlabels(n_label, n_data, pe, exec_mode):
    """The node must emit the label unchanged on out0 and pass the data through on out1."""
    label_dt, data_dt = DataType["INT8"], DataType["INT4"]
    model = make_alignlabels_modelwrapper(n_label, n_data, pe, label_dt, data_dt)
    model = model.transform(SpecializeLayers(test_fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    assert model.graph.node[0].op_type == "AlignLabels_hls"

    if exec_mode == "cppsim":
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    label_in = gen_finn_dt_tensor(label_dt, [1, n_label])
    data_in = gen_finn_dt_tensor(data_dt, [1, n_data])
    ctx = execute_onnx(model, {"lab": label_in, "dat": data_in})

    assert np.array_equal(ctx["olab"], label_in), exec_mode + ": label stream altered"
    assert np.array_equal(ctx["odat"], data_in), exec_mode + ": data passthrough altered"


def _side_channel_and_model_fifo(model):
    """Return (bypass_fifo_depth, model_path_fifo_depth) for an aligned model.

    The bypass FIFO is the StreamingFIFO fed by the DuplicateStreams and consumed
    by the AlignLabels; the model-path FIFO is the other DuplicateStreams output.
    """
    dup = model.get_nodes_by_op_type("DuplicateStreams_hls")[0]
    bypass_depth = model_depth = None
    for out in dup.output:
        cons = model.find_consumer(out)
        # a StreamingFIFO may have been inserted on the edge
        depth = 2
        if cons is not None and cons.op_type.startswith("StreamingFIFO"):
            depth = getCustomOp(cons).get_nodeattr("depth")
            cons = model.find_consumer(cons.output[0])
        if cons is not None and cons.op_type == "AlignLabels_hls":
            bypass_depth = depth
        else:
            model_depth = depth
    return bypass_depth, model_depth


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
def test_alignlabels_fifo_sizing(tmp_path):
    """The analytic sizer must size the AlignLabels bypass FIFO to ~the model
    latency (so the fork never back-pressures the model path) while leaving the
    model-path FIFO shallow."""
    import torch
    from brevitas.export import export_qonnx

    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as build_cfg
    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
    from finn.util.test import get_trained_network_and_ishape

    outdir = str(tmp_path / "align_fifo")
    net, ishape = get_trained_network_and_ishape("tfc", 2, 2)
    chkpt = str(tmp_path / "model.onnx")
    export_qonnx(net, torch.randn(ishape), chkpt)

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=outdir,
        align_labels=True,
        auto_fifo_depths=True,
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod.ANALYTIC,
        tav_generation_strategy=build_cfg.TAVGenerationMethod.TREE_MODEL,
        tav_utilization_strategy=build_cfg.TAVUtilizationMethod.CONSERVATIVE_RELAXATION,
        skip_resynth_during_fifo_sizing=True,
        target_fps=10000,
        synth_clk_period_ns=target_clk_ns,
        board="Pynq-Z1",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        save_intermediate_models=True,
        enable_build_pdb_debug=False,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        steps=[
            "step_qonnx_to_finn",
            "step_tidy_up",
            "step_streamline",
            "step_convert_to_hw",
            "step_create_dataflow_partition",
            "step_specialize_layers",
            "step_target_fps_parallelization",
            "step_apply_folding_config",
            "step_minimize_bit_width",
            "step_generate_estimate_reports",
            "step_set_fifo_depths",
        ],
    )
    assert build.build_dataflow_cfg(chkpt, cfg) == 0, "aligned build failed"

    model = ModelWrapper(outdir + "/intermediate_models/step_set_fifo_depths.onnx")
    align = model.get_nodes_by_op_type("AlignLabels_hls")
    assert len(align) == 1, "AlignLabels node missing after build"

    # the (folded) number of data transactions AlignLabels streams per frame
    frame_size = int(np.prod(getCustomOp(align[0]).get_folded_input_shape(1)[:-1]))
    # the model bottleneck period -- the buffer must span roughly this many input
    # tokens (the model latency) so the fork never back-pressures the model path.
    max_cycles = model.analysis(dataflow_performance)["max_cycles"]
    bypass_depth, model_depth = _side_channel_and_model_fifo(model)

    assert bypass_depth is not None, "no bypass FIFO on the DuplicateStreams->AlignLabels edge"
    # the sizer engaged (well beyond the default depth of 2) and sized the bypass
    # to buffer the input for the model latency, not just a shallow FIFO
    assert bypass_depth >= frame_size, f"bypass FIFO {bypass_depth} < one frame {frame_size}"
    assert (
        bypass_depth >= 0.5 * max_cycles
    ), f"bypass FIFO {bypass_depth} too small for model latency {max_cycles}"
    # the buffer FIFO, not the model-path FIFO, got the depth
    assert model_depth is not None and model_depth < bypass_depth // 4
