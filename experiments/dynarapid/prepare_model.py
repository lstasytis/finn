"""Prepare small BNN-PYNQ dataflow models (TFC / CNV) for the DynaRapid experiments.

Follows the transformation sequence of tests/end2end/test_end2end_bnn_pynq.py up to the
point where FINN would implement the design (FIFOs + DWCs inserted, every node through
IP generation), but with reduced PE/SIMD so that each layer is a small component.

The result is a dataflow-partition ONNX model whose nodes all carry generated IP; it is
the common input of both the baseline FINN flow (stitched IP + Vivado OOC P&R) and the
DynaRapid flow.

Usage:
    python prepare_model.py --topology tfc --out <dir>
    python prepare_model.py --topology tfc --out <dir> --nodes MVAU_hls_0,MVAU_hls_1
"""

import argparse
import json
import os
import time
import torch
from brevitas.export import export_qonnx
from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline.absorb as absorb
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.minimize_accumulator_width import (
    MinimizeAccumulatorWidth,
)
from finn.transformation.fpgadataflow.minimize_weight_bit_width import (
    MinimizeWeightBitWidth,
)
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.reorder import (
    MakeMaxPoolNHWC,
    MoveScalarLinearPastInvariants,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.pytorch import ToTensor
from finn.util.test import get_trained_network_and_ishape

FPGA_PART = "xck26-sfvc784-2LV-c"  # KV260
CLK_NS = 5.0

# (PE, SIMD) per MVAU, reduced from the end2end test foldings (fold_tfc / fold_cnv_small)
FOLDINGS = {
    "tfc": {
        "mvau": [(4, 7), (2, 2), (2, 2), (1, 2)],
        "thres_pe": 7,
    },
    "cnv": {
        "mvau": [(4, 3), (4, 4), (4, 4), (4, 4), (2, 4), (1, 4), (1, 2), (1, 2), (5, 1)],
        "thres_pe": 1,
    },
}


def export_and_streamline(topology, wbits, abits, work):
    """Brevitas export + tidy-up + pre/post-processing + streamlining + HW conversion."""
    (model_pt, ishape) = get_trained_network_and_ishape(topology, wbits, abits)
    chkpt = os.path.join(work, "export.onnx")
    export_qonnx(model_pt, torch.randn(ishape), chkpt, opset_version=13)
    qonnx_cleanup(chkpt, out_file=chkpt)
    model = ModelWrapper(chkpt)
    model = model.transform(ConvertQONNXtoFINN())
    model = tidy(model)
    # pre-processing (ToTensor) and post-processing (TopK), as in the end2end test
    pre = os.path.join(work, "preproc.onnx")
    export_qonnx(ToTensor(), torch.randn(ishape), pre, opset_version=13)
    qonnx_cleanup(pre, out_file=pre)
    pre_model = ModelWrapper(pre).transform(ConvertQONNXtoFINN())
    pre_model = pre_model.transform(InferShapes()).transform(FoldConstants())
    model = model.transform(MergeONNXModels(pre_model))
    model.set_tensor_datatype(model.get_first_global_in(), DataType["UINT8"])
    model = model.transform(InsertTopK(k=1))
    model = tidy(model)
    # streamline
    model = model.transform(absorb.AbsorbScalarBiasIntoMultiThreshold())
    model = model.transform(MoveScalarLinearPastInvariants())
    model = model.transform(Streamline())
    if "fc" not in topology:
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(Streamline())
    model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
    model = model.transform(InferDataLayouts())
    model = model.transform(RemoveUnusedTensors())
    # convert to HW layers
    model = model.transform(to_hw.InferBinaryMatrixVectorActivation())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_hw.InferLabelSelectLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    if "fc" not in topology:
        model = model.transform(to_hw.InferPool())
        model = model.transform(to_hw.InferConvInpGen())
        model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(absorb.AbsorbConsecutiveTransposes())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferDataLayouts())
    model = model.transform(MinimizeWeightBitWidth(datatype_only=True))
    model = model.transform(MinimizeAccumulatorWidth(datatype_only=True))
    model = model.transform(InferDataTypes())
    model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    # dataflow partition
    parent = model.transform(CreateDataflowPartition())
    sdp = getCustomOp(parent.get_nodes_by_op_type("StreamingDataflowPartition")[0])
    return ModelWrapper(sdp.get_nodeattr("model"))


def tidy(model):
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    return model


def fold(model, topology, scale=1, ram_style="auto"):
    """Apply the reduced folding. `scale` multiplies PE and SIMD of every MVAU (for the
    component-size scaling study); values are clipped to legal divisors."""
    cfg = FOLDINGS[topology]
    mvaus = model.get_nodes_by_op_type("MVAU_hls") + model.get_nodes_by_op_type("MVAU_rtl")
    for node, (pe, simd) in zip(mvaus, cfg["mvau"]):
        inst = getCustomOp(node)
        mh, mw = inst.get_nodeattr("MH"), inst.get_nodeattr("MW")
        inst.set_nodeattr("PE", largest_divisor(mh, pe * scale))
        inst.set_nodeattr("SIMD", largest_divisor(mw, simd * scale))
        inst.set_nodeattr("mem_mode", "internal_decoupled")
        inst.set_nodeattr("ram_style", ram_style)
        inst.set_nodeattr("resType", "lut")
        inst.set_nodeattr("runtime_writeable_weights", 0)
    for node in model.get_nodes_by_op_type("ConvolutionInputGenerator_rtl"):
        inst = getCustomOp(node)
        consumer = model.find_consumer(node.output[0])
        if not inst.get_nodeattr("depthwise") and consumer is not None:
            if consumer.op_type.startswith("MVAU"):
                inst.set_nodeattr("SIMD", getCustomOp(consumer).get_nodeattr("SIMD"))
        inst.set_nodeattr("ram_style", "distributed")
    for node in model.get_nodes_by_op_type("Thresholding_rtl"):
        inst = getCustomOp(node)
        inst.set_nodeattr("PE", largest_divisor(inst.get_nodeattr("NumChannels"), cfg["thres_pe"]))
        inst.set_nodeattr("runtime_writeable_weights", 0)
        inst.set_nodeattr("depth_trigger_uram", 32000)
        inst.set_nodeattr("depth_trigger_bram", 32000)
    model = model.transform(MinimizeWeightBitWidth())
    model = model.transform(MinimizeAccumulatorWidth())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(MinimizeWeightBitWidth())
    return model


def largest_divisor(n, upper):
    for d in range(min(n, upper), 0, -1):
        if n % d == 0:
            return d
    return 1


def extract_subgraph(model, node_names):
    """Keep only the named (consecutive) nodes; boundary tensors become graph I/O."""
    keep = [n for n in model.graph.node if n.name in node_names]
    assert len(keep) == len(node_names), "unknown node in %s" % node_names
    produced = {t for n in keep for t in n.output}
    consumed = {t for n in keep for t in n.input}
    # streaming inputs only (initializers such as weights/thresholds stay attached)
    ins = [
        t for n in keep for t in n.input if t not in produced and model.get_initializer(t) is None
    ]
    outs = [t for n in keep for t in n.output if t not in consumed]
    for n in list(model.graph.node):
        if n.name not in node_names:
            model.graph.node.remove(n)

    def vi(t):
        v = model.get_tensor_valueinfo(t)
        if v is None:
            v = helper.make_tensor_value_info(t, 1, model.get_tensor_shape(t))
        return v

    new_ins, new_outs = [vi(t) for t in ins], [vi(t) for t in outs]
    # boundary tensors move from value_info to the graph inputs/outputs
    for v in list(model.graph.value_info):
        if v.name in ins or v.name in outs:
            model.graph.value_info.remove(v)
    del model.graph.input[:]
    del model.graph.output[:]
    model.graph.input.extend(new_ins)
    model.graph.output.extend(new_outs)
    return model.transform(RemoveUnusedTensors())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", choices=["tfc", "cnv"], required=True)
    ap.add_argument("--wbits", type=int, default=1)
    ap.add_argument("--abits", type=int, default=2)
    ap.add_argument("--scale", type=int, default=1, help="PE/SIMD multiplier for MVAUs")
    ap.add_argument(
        "--ram-style",
        default="auto",
        help="MVAU weight memory style (auto: BRAM for large layers, as FINN would choose)",
    )
    ap.add_argument("--nodes", default="", help="comma-separated node names to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default=None, help="FPGA part (default: KV260 xck26)")
    args = ap.parse_args()
    global FPGA_PART
    if args.part:
        FPGA_PART = args.part

    os.makedirs(args.out, exist_ok=True)
    timings = {}
    t0 = time.time()
    base = os.path.join(args.out, "dataflow_streamlined.onnx")
    if os.path.isfile(base):
        model = ModelWrapper(base)
    else:
        model = export_and_streamline(args.topology, args.wbits, args.abits, args.out)
        model.save(base)
    timings["frontend_s"] = time.time() - t0

    t0 = time.time()
    model = fold(model, args.topology, args.scale, args.ram_style)
    if args.nodes:
        model = extract_subgraph(model, args.nodes.split(","))
    model = model.transform(InsertDWC())
    model = model.transform(SpecializeLayers(FPGA_PART))
    if not args.nodes:
        # shallow FIFOs between all layers; depth sizing is irrelevant for the P&R study
        model = model.transform(InsertFIFO(create_shallow_fifos=True))
        model = model.transform(SpecializeLayers(FPGA_PART))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(FPGA_PART, CLK_NS))
    model = model.transform(HLSSynthIP())
    timings["ipgen_s"] = time.time() - t0
    model.set_metadata_prop("dynarapid_fpga_part", FPGA_PART)
    model.set_metadata_prop("dynarapid_clk_ns", str(CLK_NS))
    out = os.path.join(args.out, "dataflow_ipgen.onnx")
    model.save(out)
    with open(os.path.join(args.out, "prepare_timings.json"), "w") as f:
        json.dump(timings, f, indent=2)
    print("Nodes:", [n.name for n in model.graph.node])
    print("Saved", out, timings)


if __name__ == "__main__":
    main()
