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
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
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
    FoldResult,
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
    # unrolled further, the 4-bit weight stream of an HLS MVAU (PE * SIMD * 4 bits) exceeds
    # the 8191 bits of an ap_int: the fastest folding the optimizer may choose
    fastest = getCustomOp(dataflow_model.graph.node[0])
    fastest.set_nodeattr("PE", 8)
    fastest.set_nodeattr("SIMD", 128)
    min_cycles["U200"] = fastest.get_exp_cycles() if style == "optimizer" else 1

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
def test_set_folding_padding_dwc_not_penalized_as_hls():
    # the RTL DWC pads and crops: the annealer's DWC heuristic charges a padding DWC its
    # stream widths only, not HLS_DWC_COST_PENALTY times them
    model = make_multi_fclayer_model(64, DataType["INT4"], DataType["INT2"], DataType["INT16"], 2)
    producer, consumer = (getCustomOp(n) for n in model.graph.node)
    consumer.set_nodeattr("MW", 66)
    consumer.set_nodeattr("SIMD", 11)
    opt = Optimizer(model, "test", {"max_cycles": 1})
    widths = producer.get_outstream_width() + producer.get_instream_width()
    assert opt.compute_hls_dwc_cost(model, [consumer], 1000) == widths / 1000


@pytest.mark.fpgadataflow
def test_set_folding_tighten_offers_padded_factors():
    # an MVAU folded faster than the target needs (SIMD 16: 256 of 400 cycles): the
    # tighten pass offers the padded SIMDs between the divisors 16 and 8 (8: 512 cycles)
    model = make_multi_fclayer_model(64, DataType["INT4"], DataType["INT2"], DataType["INT16"], 3)
    model = model.transform(GiveUniqueNodeNames())
    folder = SetFolding(400, platform="Pynq-Z1", folding_maximum_padding=4)
    opt = folder._make_optimizer(model, "test", {"max_cycles": 400, **folder._resource_budgets()})
    opt.generate_parameter_set()
    params = opt.params.parameters
    values = [16 if p.name == "SIMD" else p.possible_values[0] for p in params]
    opt.params.add_all_params_to_index_list()
    opt.params.set_values(values)
    opt.params.apply_updates(attrs=opt.PARAMETERS_TO_APPLY)
    i = next(j for j, p in enumerate(params) if p.node_index == 1 and p.name == "SIMD")
    moved = folder._slower_values(opt, values, i, 400)
    # 65 / 13 and 66 / 11 channels: 320 and 384 cycles
    assert {m[i] for m in moved} == {11, 13}
    assert all(m[:i] + m[i + 1 :] == values[:i] + values[i + 1 :] for m in moved)
    assert params[i].meta_value == 16


@pytest.mark.fpgadataflow
def test_set_folding_padded_kept_when_cheaper():
    # among foldings meeting the target the padded one is kept when it costs less, even
    # at a higher peak utilization (zero-padded weights can cost a BRAM and save LUTs)
    folder = SetFolding(100, platform="Pynq-Z1")
    folder._budgets = folder._resource_budgets()
    plain = FoldResult(None, True, {"LUT": 4000.0, "BRAM_18K": 100.0}, 100, [], 1.0)
    padded = FoldResult(None, True, {"LUT": 1000.0, "BRAM_18K": 101.0}, 100, [], 0.9)
    assert folder._better(plain, padded, 100, False) is padded
    assert folder._better(plain, padded._replace(cost=1.1), 100, False) is plain
    assert folder._better(plain, padded._replace(achieved=101), 100, False) is plain


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


def _hw_node(op_type, inputs, outputs, **attrs):
    return helper.make_node(
        op_type,
        inputs,
        outputs,
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        **attrs,
    )


@pytest.mark.fpgadataflow
def test_set_folding_swg_pair_in_branching_graph():
    # two SWG -> MVAU branches, interleaved: an SWG is paired with its consumer,
    # not with the next node in the graph (here the other branch's SWG)
    ch, mh, ifm, k = 8, 16, 6, 3
    ofm = ifm - k + 1
    swgs, mvaus, inputs, outputs, inner = [], [], [], [], []
    for b in "ab":
        inputs.append(
            helper.make_tensor_value_info("in_" + b, TensorProto.FLOAT, [1, ifm, ifm, ch])
        )
        outputs.append(
            helper.make_tensor_value_info("out_" + b, TensorProto.FLOAT, [1, ofm, ofm, mh])
        )
        inner.append(
            helper.make_tensor_value_info("win_" + b, TensorProto.FLOAT, [1, ofm, ofm, k * k * ch])
        )
        swgs.append(
            _hw_node(
                "ConvolutionInputGenerator_rtl",
                ["in_" + b],
                ["win_" + b],
                ConvKernelDim=[k, k],
                IFMChannels=ch,
                IFMDim=[ifm, ifm],
                OFMDim=[ofm, ofm],
                SIMD=1,
                Stride=[1, 1],
                Dilation=[1, 1],
                inputDataType="INT4",
                outputDataType="INT4",
            )
        )
        mvaus.append(
            _hw_node(
                "MVAU_rtl",
                ["win_" + b, "w_" + b],
                ["out_" + b],
                MW=k * k * ch,
                MH=mh,
                SIMD=1,
                PE=1,
                numInputVectors=[1, ofm, ofm],
                inputDataType="INT4",
                weightDataType="INT4",
                outputDataType="INT32",
                noActivation=1,
                mem_mode="internal_decoupled",
            )
        )
    graph = helper.make_graph(swgs + mvaus, "branches", inputs, outputs, value_info=inner)
    model = ModelWrapper(qonnx_make_model(graph))
    for b in "ab":
        model.set_initializer("w_" + b, np.ones((k * k * ch, mh), dtype=np.float32))
        for name, dt in (("in_", "INT4"), ("win_", "INT4"), ("w_", "INT4"), ("out_", "INT32")):
            model.set_tensor_datatype(name + b, DataType[dt])
    target = ofm * ofm * 4
    model = model.transform(SetFolding(target, platform="U250", folding_effort=20, seed=0))
    assert max(model.analysis(exp_cycles_per_layer).values()) <= target


@pytest.mark.parametrize("padding", [0, 4])
@pytest.mark.fpgadataflow
def test_set_folding_conv_mvau_pe_padded(padding):
    # a conv MVAU's PE is folded over its (padded) output channels like a dense one's
    ch, mh, ifm, k = 8, 14, 6, 3
    ofm = ifm - k + 1
    swg = _hw_node(
        "ConvolutionInputGenerator_rtl",
        ["inp"],
        ["win"],
        ConvKernelDim=[k, k],
        IFMChannels=ch,
        IFMDim=[ifm, ifm],
        OFMDim=[ofm, ofm],
        SIMD=1,
        Stride=[1, 1],
        Dilation=[1, 1],
        inputDataType="INT4",
        outputDataType="INT4",
    )
    mvau = _hw_node(
        "MVAU_rtl",
        ["win", "w"],
        ["outp"],
        MW=k * k * ch,
        MH=mh,
        SIMD=1,
        PE=1,
        numInputVectors=[1, ofm, ofm],
        inputDataType="INT4",
        weightDataType="INT4",
        outputDataType="INT32",
        noActivation=1,
        mem_mode="internal_decoupled",
    )
    graph = helper.make_graph(
        [swg, mvau],
        "conv",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, ifm, ifm, ch])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, ofm, ofm, mh])],
        value_info=[
            helper.make_tensor_value_info("win", TensorProto.FLOAT, [1, ofm, ofm, k * k * ch])
        ],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w", np.ones((k * k * ch, mh), dtype=np.float32))
    for name, dt in (("inp", "INT4"), ("win", "INT4"), ("w", "INT4"), ("outp", "INT32")):
        model.set_tensor_datatype(name, DataType[dt])
    opt = Optimizer(model, "test", {"max_cycles": 1}, padding=padding, pad_io_nodes=True)
    opt.generate_parameter_set()
    pe = next(p for p in opt.params.parameters if p.name == "PE")
    channels = {v: real[0].bound_value for v, real in zip(pe.possible_values, pe.real_values)}
    if padding:
        # 15 / 3, 15 / 5 and 16 / 4 channels between the divisors of 14
        assert channels == {1: 14, 2: 14, 3: 15, 4: 16, 5: 15, 7: 14, 14: 14}
    else:
        assert channels == {1: 14, 2: 14, 7: 14, 14: 14}


@pytest.mark.fpgadataflow
def test_set_folding_dynamic_mvau():
    # an MVAU with streamed weights (mem_mode="dynamic") has no weight initializer:
    # it is folded, but neither its weights nor its MW / MH are padded
    mw, mh, vecs = 30, 18, 8
    graph = helper.make_graph(
        [
            _hw_node(
                "MVAU_rtl",
                ["inp", "wfm"],
                ["outp"],
                MW=mw,
                MH=mh,
                SIMD=1,
                PE=1,
                numInputVectors=[1, vecs],
                inputDataType="INT4",
                weightDataType="INT4",
                outputDataType="INT32",
                noActivation=1,
                mem_mode="dynamic",
            )
        ],
        "dynamic_mvau",
        [
            helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, vecs, mw]),
            helper.make_tensor_value_info("wfm", TensorProto.FLOAT, [mw, mh]),
        ],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, vecs, mh])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    for name, dt in (("inp", "INT4"), ("wfm", "INT4"), ("outp", "INT32")):
        model.set_tensor_datatype(name, DataType[dt])
    folder = SetFolding(
        vecs * 6,
        platform="U250",
        folding_effort=20,
        folding_maximum_padding=4,
        folding_pad_io_nodes=True,
        seed=0,
    )
    model = model.transform(folder)
    inst = getCustomOp(model.graph.node[0])
    assert (inst.get_nodeattr("MW"), inst.get_nodeattr("MH")) == (mw, mh)
    assert model.get_tensor_shape("wfm") == [mw, mh]
    assert inst.get_exp_cycles() <= vecs * 6


# Versal: RTL Requant/ElementwiseMul/HWSoftmax, Zynq UltraScale+: HLS variants
@pytest.mark.parametrize("part", ["xcvc1902-vsva2197-2MP-e-S", "xczu7ev-ffvc1156-2-e"])
@pytest.mark.parametrize("style", ["naive", "optimizer"])
@pytest.mark.fpgadataflow
def test_set_folding_requant_elementwise_softmax(part, style):
    # Quant (8-bit) -> Mul (float per-channel scale) -> Softmax
    ishape = [1, 8, 64]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    nodes = [
        helper.make_node(
            "Quant",
            ["inp", "scale", "zeropt", "bitwidth"],
            ["quant_out"],
            domain="qonnx.custom_op.general",
            signed=1,
            narrow=0,
            rounding_mode="ROUND",
        ),
        helper.make_node("Mul", ["quant_out", "mul_param"], ["mul_out"]),
        helper.make_node("Softmax", ["mul_out"], ["outp"], axis=-1),
    ]
    model = ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "rq_mul_sm", [inp], [outp])))
    model.set_initializer("scale", np.array(1.0, dtype=np.float32))
    model.set_initializer("zeropt", np.array(0.0, dtype=np.float32))
    model.set_initializer("bitwidth", np.array(8.0, dtype=np.float32))
    model.set_initializer("mul_param", gen_finn_dt_tensor(DataType["FLOAT32"], [ishape[-1]]))
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    model = model.transform(to_hw.InferRequantLayer())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(to_hw.InferHWSoftmax())
    model = model.transform(SpecializeLayers(part))
    model = model.transform(GiveUniqueNodeNames())
    impl = "rtl" if part.startswith("xcvc") else "hls"
    op_types = ["Requant_" + impl, "ElementwiseMul_" + impl, "HWSoftmax_" + impl]
    assert sorted(n.op_type for n in model.graph.node) == sorted(op_types)

    # target below the unfolded cycle count, so all layers have to be parallelized
    target_cycles = min(getCustomOp(n).get_exp_cycles() for n in model.graph.node) // 4
    model = model.transform(
        SetFolding(target_cycles_per_frame=target_cycles, two_pass_relaxation=False, style=style)
    )
    for node in model.graph.node:
        inst = getCustomOp(node)
        attr = "SIMD" if node.op_type.startswith("HWSoftmax") else "PE"
        fold = inst.get_nodeattr(attr)
        assert fold > 1 and ishape[-1] % fold == 0, f"{node.name}: {attr}={fold}"


@pytest.mark.parametrize("part", ["xczu7ev-ffvc1156-2-e", "xcvc1902-vsva2197-2MP-e-S"])
@pytest.mark.fpgadataflow
def test_set_folding_no_embedded_uram_off_versal(part):
    # an elementwise op's embedded constants may only go to URAM on Versal
    ishape = [1, 8, 64]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    node = helper.make_node("Mul", ["inp", "mul_param"], ["outp"])
    model = ModelWrapper(qonnx_make_model(helper.make_graph([node], "mul", [inp], [outp])))
    model.set_tensor_datatype("inp", DataType["INT8"])
    model.set_initializer("mul_param", gen_finn_dt_tensor(DataType["INT8"], [ishape[-1]]))
    model.set_tensor_datatype("mul_param", DataType["INT8"])
    model = model.transform(InferShapes()).transform(InferDataTypes())
    model = model.transform(to_hw.InferElementwiseBinaryOperation())
    model = model.transform(SpecializeLayers(part)).transform(GiveUniqueNodeNames())
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("mem_mode") == "internal_embedded"
    opt = Optimizer(model, "t", {"max_cycles": 1}, fpgapart=part)
    meta = opt._make_resource_type_meta(inst, 0, "ram_style")
    offered = meta.possible_values if meta is not None else []
    if part.startswith("xczu"):
        assert "ultra" not in offered


@pytest.mark.parametrize("style", ["naive", "optimizer"])
@pytest.mark.fpgadataflow
def test_set_folding_split_concat_simd(style):
    # Split into four streams of 8 channels and Concat them again: both fold SIMD,
    # to a value dividing every stream's channel count
    ishape, chans = [1, 64, 32], [8, 8, 8, 8]
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ishape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ishape)
    parts = ["s%d" % i for i in range(len(chans))]
    nodes = [
        helper.make_node("Split", ["inp", "split"], parts, axis=-1),
        helper.make_node("Concat", parts, ["outp"], axis=-1),
    ]
    vi = [helper.make_tensor_value_info(p, TensorProto.FLOAT, [1, 64, 8]) for p in parts]
    graph = helper.make_graph(nodes, "split_concat", [inp], [outp], value_info=vi)
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("split", np.array(chans, dtype=np.int64))
    for t in ["inp", "outp"] + parts:
        model.set_tensor_datatype(t, DataType["INT8"])
    model = model.transform(to_hw.InferSplitLayer())
    model = model.transform(to_hw.InferConcatLayer())
    model = model.transform(SpecializeLayers("xczu7ev-ffvc1156-2-e"))
    model = model.transform(GiveUniqueNodeNames())
    assert [n.op_type for n in model.graph.node] == ["StreamingSplit_hls", "StreamingConcat_hls"]
    target = min(getCustomOp(n).get_exp_cycles() for n in model.graph.node) // 4
    model = model.transform(
        SetFolding(target_cycles_per_frame=target, two_pass_relaxation=False, style=style)
    )
    for node in model.graph.node:
        inst = getCustomOp(node)
        simd = inst.get_nodeattr("SIMD")
        assert simd > 1 and 8 % simd == 0, f"{node.name}: SIMD={simd}"
        assert inst.get_exp_cycles() <= target


def _mvau_hls_model(mw, mh, wdt, mem_mode):
    """A single dense MVAU_hls (no activation) with random ``wdt`` weights."""
    node = helper.make_node(
        "MVAU_hls",
        ["inp", "w"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType="INT4",
        weightDataType=wdt,
        outputDataType="INT32",
        noActivation=1,
        mem_mode=mem_mode,
    )
    graph = helper.make_graph(
        [node],
        "mvau_hls",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, mw])],
        [helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, mh])],
    )
    model = ModelWrapper(qonnx_make_model(graph))
    model.set_initializer("w", gen_finn_dt_tensor(DataType[wdt], (mw, mh)))
    for name, dt in (("inp", "INT4"), ("w", wdt), ("outp", "INT32")):
        model.set_tensor_datatype(name, DataType[dt])
    return model.transform(GiveUniqueNodeNames())


# target 1: as fast as possible; None: maximize (resnet50 on V80 got PE=32, SIMD=49 at 8 bits)
@pytest.mark.parametrize("target", [1, None])
@pytest.mark.fpgadataflow
def test_set_folding_hls_mvau_weight_stream_width(target):
    # unrolled, the weight stream of this MVAU_hls is 32 * 64 * 8 = 16384 bits wide, beyond
    # the 8191 bits of an HLS ap_int: the optimizer must stay within PE * SIMD * 8 <= 8191
    mw, mh = 64, 32
    model = _mvau_hls_model(mw, mh, "INT8", "internal_decoupled")
    opt = Optimizer(model, "t", {"max_cycles": 1})
    opt.generate_parameter_set()
    # each value alone is offered (8191 bits are exceeded only by the product)
    assert {p.name: max(p.possible_values) for p in opt.params.parameters} == {
        "SIMD": mw,
        "PE": mh,
    }
    model = model.transform(SetFolding(target, platform="U250", folding_effort=20, seed=0))
    inst = getCustomOp(model.graph.node[0])
    pe, simd = inst.get_nodeattr("PE"), inst.get_nodeattr("SIMD")
    assert pe * simd * 8 <= Optimizer.HLS_MAX_WIDTH
    assert not Optimizer.breaks_hls_limits(inst)
    # and as fast as the limit allows: PE * SIMD = 512 (1024 would be 8192 bits)
    assert pe * simd == 512


@pytest.mark.parametrize(
    "mw, mh, wdt, min_simd",
    [
        # 512 x 128 embedded weights: at SIMD=1 the initializer has 65536 entries
        (512, 128, "INT2", 2),
        # MVAU_hls ipgen requires SIMD >= MW / 1024
        (2048, 8, "INT2", 2),
    ],
)
@pytest.mark.fpgadataflow
def test_set_folding_hls_mvau_simd_limits(mw, mh, wdt, min_simd):
    # a slow target would fold SIMD to 1; the HLS limits keep it at min_simd or above
    model = _mvau_hls_model(mw, mh, wdt, "internal_embedded")
    opt = Optimizer(model, "t", {"max_cycles": 1})
    opt.generate_parameter_set()
    simd = next(p for p in opt.params.parameters if p.name == "SIMD")
    assert min(simd.possible_values) == min_simd
    model = model.transform(SetFolding(mw * mh, platform="U250", folding_effort=20, seed=0))
    inst = getCustomOp(model.graph.node[0])
    assert inst.get_nodeattr("SIMD") >= min_simd
    assert not Optimizer.breaks_hls_limits(inst)
