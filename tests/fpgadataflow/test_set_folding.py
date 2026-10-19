# Copyright (c) 2020, Xilinx
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

import pytest

import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.set_folding import (
    Optimizer,
    SetFolding,
    analytic_fifo_sizer,
    insert_and_size_fifos,
)
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir, part_map
from finn.util.platforms import DEFAULT_RES_LIMITS, platforms
from finn.util.test import load_test_checkpoint_or_skip


def make_multi_fclayer_model(ch, wdt, adt, tdt, nnodes):
    W = np.random.randint(wdt.min(), wdt.max() + 1, size=(ch, ch))
    W = W.astype(np.float32)

    T = np.random.randint(tdt.min(), tdt.max() + 1, size=(ch, 2 ** adt.bitwidth() - 1))
    T = T.astype(np.float32)

    tensors = []
    tensors.append(helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch]))
    for i in range(1, nnodes):
        inter = helper.make_tensor_value_info("inter_" + str(i), TensorProto.FLOAT, [1, ch])
        tensors.append(inter)
    tensors.append(helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch]))

    FCLayer_nodes = []
    for i in range(nnodes):
        pe = 1
        simd = 1
        FCLayer_nodes += [
            helper.make_node(
                "MVAU_hls",
                [tensors[i].name, "weights_" + str(i), "thresh_" + str(i)],
                [tensors[i + 1].name],
                domain="finn.custom_op.fpgadataflow.hls",
                backend="fpgadataflow",
                MW=ch,
                MH=ch,
                SIMD=simd,
                PE=pe,
                inputDataType=adt.name,
                weightDataType=wdt.name,
                outputDataType=adt.name,
                ActVal=0,
                binaryXnorMode=0,
                noActivation=0,
            )
        ]

    graph = helper.make_graph(
        nodes=FCLayer_nodes,
        name="fclayer_graph",
        inputs=[tensors[0]],
        outputs=[tensors[-1]],
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", adt)
    model.set_tensor_datatype("outp", adt)

    for i in range(1, nnodes + 1):
        if tensors[i].name != "outp":
            model.graph.value_info.append(tensors[i])
        model.set_initializer("weights_" + str(i - 1), W)
        model.set_initializer("thresh_" + str(i - 1), T)
        model.set_tensor_datatype("weights_" + str(i - 1), wdt)
        model.set_tensor_datatype("thresh_" + str(i - 1), tdt)

    return model


# desired frames per second
@pytest.mark.parametrize("target_fps", [30, 10**5, 10**7])
# target chip or board
@pytest.mark.parametrize("platform", ["AUP-ZU3_8GB", "Ultra96", "U200"])
@pytest.mark.parametrize("style", ["naive", "optimizer"])
@pytest.mark.fpgadataflow
def test_set_folding(target_fps, platform, style):
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 5)

    model = model.transform(GiveUniqueNodeNames())
    parent_model = model.transform(CreateDataflowPartition())
    sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
    sdp_node = getCustomOp(sdp_node)
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = load_test_checkpoint_or_skip(dataflow_model_filename)

    clk_ns = 5
    target_cycles_per_frame = int((10**9 / clk_ns) / target_fps)
    dataflow_model = dataflow_model.transform(
        SetFolding(target_cycles_per_frame, platform=platform, style=style, folding_effort=200)
    )

    exp_cycles_dict = dataflow_model.analysis(exp_cycles_per_layer)
    achieved_cycles_per_frame = max(exp_cycles_dict.values())

    min_cycles = dict()
    min_cycles["AUP-ZU3_8GB"] = 64
    min_cycles["Ultra96"] = 64
    # SIMD <= 8 (mvau_wwidth_max): 128 / 8 cycles plus the MVAU_hls per-frame overhead
    min_cycles["U200"] = 16 + 7

    assert achieved_cycles_per_frame <= max(
        min_cycles[platform], target_cycles_per_frame
    ), "Folding target not met"


def _optimizer_fold(nnodes=5, **kwargs):
    """Fold an nnodes-deep MVAU model with the optimizer style (defaults suited
    to fast, deterministic unit tests)."""
    model = make_multi_fclayer_model(
        128, DataType["INT4"], DataType["INT2"], DataType["INT16"], nnodes
    )
    model = model.transform(GiveUniqueNodeNames())
    kwargs.setdefault("platform", "U250")
    kwargs.setdefault("style", "optimizer")
    kwargs.setdefault("folding_effort", 50)
    kwargs.setdefault("seed", 0)
    return model.transform(SetFolding(**kwargs))


@pytest.mark.fpgadataflow
def test_set_folding_maximize_throughput():
    # target_cycles_per_frame=None => maximize throughput within the resource
    # budget (no user-supplied ceiling). On a large device (U250) the optimizer
    # must actually fold -- i.e. run strictly faster than the fully unfolded MVAU
    # stack (128*128 cycles) -- while keeping resource usage within budget.
    unfolded = 128 * 128
    model = _optimizer_fold(target_cycles_per_frame=None)
    achieved = max(model.analysis(exp_cycles_per_layer).values())
    assert 0 < achieved < unfolded
    budget_lut = DEFAULT_RES_LIMITS[0] * sum(
        r["LUT"] for r in platforms["U250"](1).resource_count_dict.values()
    )
    total_lut = sum(
        getCustomOp(n).node_res_estimation(part_map["U250"]).get("LUT", 0) for n in model.graph.node
    )
    assert total_lut <= budget_lut


@pytest.mark.fpgadataflow
def test_set_folding_naive_requires_target():
    # the naive folder has no maximize-throughput mode, so a None target (which
    # means "maximize" for the optimizer) must be rejected with a clear error
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 2)
    model = model.transform(GiveUniqueNodeNames())
    with pytest.raises(ValueError):
        model.transform(SetFolding(target_cycles_per_frame=None, style="naive"))


@pytest.mark.parametrize("style", ["optimized", "Optimizer", "greedy", ""])
def test_set_folding_rejects_unknown_style(style):
    # An unrecognized style used to fall through to the optimizer, so a
    # misspelling ("optimized" for "optimizer") silently ran a folder the caller
    # never asked for and the build still succeeded.
    model = make_multi_fclayer_model(128, DataType["INT4"], DataType["INT2"], DataType["INT16"], 2)
    model = model.transform(GiveUniqueNodeNames())
    with pytest.raises(ValueError, match="unknown SetFolding style"):
        model.transform(SetFolding(target_cycles_per_frame=1000, style=style))


@pytest.mark.parametrize("prefer_memory", ["bram", "uram"])
@pytest.mark.fpgadataflow
def test_set_folding_prefer_memory(prefer_memory):
    # the always-on resource-type pass must still meet the throughput target and
    # must never pick the mis-estimated "distributed" ram_style (excluded so the
    # BRAM<->URAM trade stays honest)
    model = _optimizer_fold(target_cycles_per_frame=2000, prefer_memory=prefer_memory)
    for node in model.graph.node:
        inst = getCustomOp(node)
        if "ram_style" in inst.get_nodeattr_types():
            assert inst.get_nodeattr("ram_style") != "distributed"
    assert max(model.analysis(exp_cycles_per_layer).values()) <= 2000


@pytest.mark.xfail(
    strict=True,
    reason="FINN's node_res_estimation is currently blind to resType on RTL "
    "MVAU/VVAU, so the optimizer's LUT<->DSP trade is a no-op there. Remove this "
    "xfail once the estimator scores resType (the optimizer already tunes it "
    "wherever the estimate responds).",
)
@pytest.mark.fpgadataflow
def test_resType_changes_rtl_mvau_estimate():
    # a folded RTL MVAU *should* estimate differently for resType "lut" vs "dsp";
    # today it does not, which is exactly why the optimizer skips this inert lever
    ch = 256
    W = np.random.randint(-8, 8, size=(ch, ch)).astype(np.float32)
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ch])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ch])
    node = helper.make_node(
        "MVAU_rtl",
        ["inp", "w"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=ch,
        MH=ch,
        SIMD=16,
        PE=16,
        inputDataType="INT4",
        weightDataType="INT4",
        outputDataType="INT16",
        noActivation=1,
    )
    model = ModelWrapper(qonnx_make_model(helper.make_graph([node], "g", [inp], [outp])))
    model.set_initializer("w", W)
    model.set_tensor_datatype("w", DataType["INT4"])
    model.set_tensor_datatype("inp", DataType["INT4"])
    model = model.transform(GiveUniqueNodeNames())
    inst = getCustomOp(model.graph.node[0])
    part = part_map["U250"]
    inst.set_nodeattr("resType", "lut")
    lut_est = inst.node_res_estimation(part)
    inst.set_nodeattr("resType", "dsp")
    dsp_est = inst.node_res_estimation(part)
    assert lut_est != dsp_est, "resType does not affect the RTL MVAU estimate"


def _with_dwcs(model, platform):
    """The graph the build carries: DWCs inserted and specialized, cycles annotated."""
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(part_map[platform]))
    model = model.transform(GiveUniqueNodeNames())
    return model.transform(AnnotateCycles())


def _resources(model, platform):
    per_node = model.analysis(lambda m: res_estimation(m, part_map[platform]))
    return {r: sum(res.get(r, 0) for res in per_node.values()) for r in ("LUT", "BRAM_18K")}


@pytest.mark.fpgadataflow
def test_set_folding_dwc_cycles_estimate():
    # the optimizer's cost model must see the DWCs the build inserts the way
    # StreamingDataWidthConverter counts them, including a non-integer width ratio
    model = make_multi_fclayer_model(48, DataType["INT4"], DataType["INT2"], DataType["INT16"], 3)
    model = model.transform(GiveUniqueNodeNames())
    for node, (simd, pe) in zip(model.graph.node, [(8, 3), (4, 6), (12, 2)]):
        getCustomOp(node).set_nodeattr("SIMD", simd)
        getCustomOp(node).set_nodeattr("PE", pe)
    estimated = {
        node.name: Optimizer.dwc_cycles(model, getCustomOp(node)) for node in model.graph.node
    }
    inserted = _with_dwcs(model, "U250")
    dwcs = [n for n in inserted.graph.node if n.op_type.startswith("StreamingDataWidthConverter")]
    assert len(dwcs) == 2
    for dwc in dwcs:
        consumer = inserted.find_consumer(dwc.output[0])
        assert estimated[consumer.name] == getCustomOp(dwc).get_exp_cycles()
    assert estimated["MVAU_hls_0"] == 0


@pytest.mark.parametrize("target", [150, 600])
@pytest.mark.fpgadataflow
def test_set_folding_optimizer_meets_target_with_dwcs(target):
    # the target is met on the graph the build carries, DWCs included, not only
    # node by node; and strictly, without a relaxation
    model = make_multi_fclayer_model(48, DataType["INT4"], DataType["INT2"], DataType["INT16"], 4)
    model = model.transform(GiveUniqueNodeNames())
    folder = SetFolding(target, platform="U250", style="optimizer", folding_effort=50, seed=0)
    model = model.transform(folder)
    built = _with_dwcs(model, "U250")
    assert built.analysis(dataflow_performance)["max_cycles"] <= target
    assert folder.last_search_report["achieved_cycles_per_frame"] <= target
    assert folder.last_search_report["fits_budget"]


@pytest.mark.parametrize("target", [150, None])
@pytest.mark.fpgadataflow
def test_set_folding_padding_never_worse(target):
    # the padded search space contains the unpadded one, so with padding the
    # result may not be worse: same target met at no higher cost, or (maximize)
    # no more cycles within the budget
    if not hasattr(StreamingDataWidthConverter, "pads_or_crops"):
        pytest.skip("DWC padding/cropping not available on this tree")
    results = {}
    for padding in (0, 4):
        model = make_multi_fclayer_model(
            40, DataType["INT4"], DataType["INT2"], DataType["INT16"], 4
        )
        model = model.transform(GiveUniqueNodeNames())
        folder = SetFolding(
            target,
            platform="Pynq-Z1",
            style="optimizer",
            folding_effort=50,
            folding_maximum_padding=padding,
            seed=0,
        )
        built = _with_dwcs(model.transform(folder), "Pynq-Z1")
        results[padding] = (
            built.analysis(dataflow_performance)["max_cycles"],
            _resources(built, "Pynq-Z1"),
            folder.last_search_report["fits_budget"],
        )
    (plain_cycles, plain_res, plain_fits), (pad_cycles, pad_res, pad_fits) = results[0], results[4]
    assert plain_fits and pad_fits
    if target is None:
        assert pad_cycles <= plain_cycles
    else:
        assert plain_cycles <= target and pad_cycles <= target
        assert pad_res["LUT"] <= plain_res["LUT"] * 1.02


@pytest.mark.fpgadataflow
def test_set_folding_bottleneck_is_unnamed_dwc():
    # the DWCs InsertDWC adds are unnamed: the slowest one must still be found
    # (4 -> 6 bits, 24 + 16 words), and with it both its neighbours
    model = make_multi_fclayer_model(48, DataType["INT4"], DataType["INT2"], DataType["INT16"], 4)
    model = model.transform(GiveUniqueNodeNames())
    for node, (simd, pe) in zip(model.graph.node, [(48, 2), (3, 48), (48, 48), (24, 48)]):
        getCustomOp(node).set_nodeattr("SIMD", simd)
        getCustomOp(node).set_nodeattr("PE", pe)
    scored = model.transform(InsertDWC()).transform(SpecializeLayers(part_map["U250"]))
    scored = scored.transform(AnnotateCycles())
    dwcs = [n for n in scored.graph.node if n.op_type.startswith("StreamingDataWidthConverter")]
    assert [n.name for n in dwcs] == ["", ""]
    assert [getCustomOp(n).get_nodeattr("cycles_estimate") for n in dwcs] == [40, 2]
    assert SetFolding._bottleneck_nodes(scored) == {"MVAU_hls_0", "MVAU_hls_1"}


def _flatten_fc_model(rows=4, ch=8):
    """Four MVAUs, the third one a dense layer over the flattened rows of the second."""
    mvaus = [(rows, 2 * ch), (rows, ch), (1, rows * ch), (1, ch)]  # (input rows, MW)
    shapes = [[1, rows, 2 * ch], [1, rows, ch], [1, rows, ch], [1, ch], [1, ch]]
    names = ["inp", "t0", "t1", "t2", "outp"]
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        "MVAU_hls",
                        [names[i], f"w{i}"],
                        [names[i + 1]],
                        domain="finn.custom_op.fpgadataflow.hls",
                        backend="fpgadataflow",
                        MW=mw,
                        MH=ch,
                        SIMD=1,
                        PE=1,
                        numInputVectors=[1, n] if n > 1 else [1],
                        inputDataType="INT4",
                        weightDataType="INT4",
                        outputDataType="INT4",
                        noActivation=1,
                        binaryXnorMode=0,
                    )
                    for i, (n, mw) in enumerate(mvaus)
                ],
                "flatten_fc",
                [helper.make_tensor_value_info("inp", TensorProto.FLOAT, shapes[0])],
                [helper.make_tensor_value_info("outp", TensorProto.FLOAT, shapes[-1])],
                value_info=[
                    helper.make_tensor_value_info(n, TensorProto.FLOAT, s)
                    for n, s in zip(names[1:-1], shapes[1:-1])
                ],
            )
        )
    )
    for i, (_, mw) in enumerate(mvaus):
        model.set_initializer(f"w{i}", np.ones((mw, ch), dtype=np.float32))
        model.set_tensor_datatype(f"w{i}", DataType["INT4"])
    model.set_tensor_datatype("inp", DataType["INT4"])
    return model.transform(GiveUniqueNodeNames())


@pytest.mark.fpgadataflow
def test_set_folding_no_padding_next_to_reshape():
    # a DWC pads or crops rows, the same on both sides: across the flatten in front
    # of a dense layer neither neighbour may be padded
    model = _flatten_fc_model()
    opt = Optimizer(model, "test", {"max_cycles": 1}, padding=4, pad_io_nodes=True)
    assert [opt._max_padding_for_node(i) for i in range(4)] == [4, 0, 0, 4]
    folder = SetFolding(
        32, platform="Pynq-Z1", folding_effort=10, folding_maximum_padding=4, seed=0
    )
    folded = model.transform(folder)
    assert getCustomOp(folded.graph.node[1]).get_nodeattr("MH") == 8
    assert getCustomOp(folded.graph.node[2]).get_nodeattr("MW") == 32
    _with_dwcs(folded, "Pynq-Z1")


def _fifo_lut(model, platform):
    per_node = model.analysis(lambda m: res_estimation(m, part_map[platform]))
    return sum(
        res.get("LUT", 0)
        for name, res in per_node.items()
        if model.get_node_from_name(name).op_type.startswith("StreamingFIFO")
    )


@pytest.mark.fpgadataflow
def test_set_folding_fifo_estimate_tracks_sizer():
    # the FIFO resources the cost model estimates for a folding must be of the
    # order of what the analytic sizer makes of it (its LUTs are mostly width)
    if analytic_fifo_sizer() is None:
        pytest.skip("no analytic FIFO sizer on this tree")
    model = make_multi_fclayer_model(48, DataType["INT4"], DataType["INT2"], DataType["INT16"], 4)
    model = model.transform(GiveUniqueNodeNames())
    for node, (simd, pe) in zip(model.graph.node, [(8, 3), (6, 6), (12, 2), (16, 48)]):
        getCustomOp(node).set_nodeattr("SIMD", simd)
        getCustomOp(node).set_nodeattr("PE", pe)
    budgets = {"max_cycles": 150, "LUT": 1e5, "BRAM_18K": 1e3, "URAM": 1e2, "DSP": 1e3}
    opt = Optimizer(model, "x", budgets, fpgapart=part_map["U250"])
    estimate = opt.fifo_resources(model, opt.graph_nodes())
    scored = model.transform(InsertDWC()).transform(SpecializeLayers(part_map["U250"]))
    work_dir = make_build_dir("test_fifo_estimate_")
    sized = insert_and_size_fifos(scored, part_map["U250"], work_dir)
    assert sized is not None
    real = _fifo_lut(sized, "U250")
    assert real > 0 and 0.5 * real <= estimate["LUT"] <= 2 * real


@pytest.mark.fpgadataflow
def test_set_folding_fifo_aware_mode():
    # with enable_folding_fifo_heuristic the verdict carries the sized FIFOs'
    # resources, the search stays synthesis-free and the target is still met
    if analytic_fifo_sizer() is None:
        pytest.skip("no analytic FIFO sizer on this tree")
    results = {}
    for fifo_aware in (False, True):
        model = make_multi_fclayer_model(
            48, DataType["INT4"], DataType["INT2"], DataType["INT16"], 4
        )
        model = model.transform(GiveUniqueNodeNames())
        folder = SetFolding(
            150,
            platform="U250",
            style="optimizer",
            folding_effort=50,
            seed=0,
            enable_folding_fifo_heuristic=fifo_aware,
        )
        model = model.transform(folder)
        built = _with_dwcs(model, "U250")
        assert built.analysis(dataflow_performance)["max_cycles"] <= 150
        results[fifo_aware] = folder.last_search_report["resources"]["LUT"]
    # the sized FIFOs' LUTs are part of the FIFO-aware verdict
    assert results[True] > 0
    assert not any(
        n.op_type.startswith("StreamingFIFO") for n in model.graph.node
    ), "the returned folding carries no FIFOs"


def test_insert_and_size_fifos_unsizeable_graph():
    # a DWC with a non-integer width ratio has no tree model: the in-loop sizer
    # must decline (None) rather than synthesize anything
    if analytic_fifo_sizer() is None:
        pytest.skip("no analytic FIFO sizer on this tree")
    model = make_multi_fclayer_model(48, DataType["INT4"], DataType["INT2"], DataType["INT16"], 2)
    model = model.transform(GiveUniqueNodeNames())
    getCustomOp(model.graph.node[0]).set_nodeattr("PE", 3)
    getCustomOp(model.graph.node[1]).set_nodeattr("SIMD", 4)
    scored = model.transform(InsertDWC()).transform(SpecializeLayers(part_map["U250"]))
    with pytest.warns(UserWarning, match="no tree model"):
        assert insert_and_size_fifos(scored, part_map["U250"], make_build_dir("test_fifo_")) is None
