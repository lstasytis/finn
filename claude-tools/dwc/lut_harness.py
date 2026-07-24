#!/usr/bin/env python3
"""LUT-measurement harness for the generalized (padding) DWC.

Builds a single StreamingDataWidthConverter node, runs SpecializeLayers +
PrepareIP + HLSSynthIP, and reads the Vitis HLS csynth LUT/FF estimate.
Also supports the plain (non-generalized) HLS DWC and the RTL DWC for
reference comparison.

Usage:
  python lut_harness.py                       # default width sweep
  python lut_harness.py --inw 512 --outw 256  # single point
"""
import argparse
import json
import numpy as np
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.analysis.fpgadataflow.hls_synth_res_estimation import hls_synth_res_estimation

TEST_PART = "xc7z020clg400-1"


def make_dwc(in_shape, out_shape, inWidth, outWidth, dt, style):
    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, in_shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, out_shape)
    node = helper.make_node(
        "StreamingDataWidthConverter",
        ["inp"],
        ["outp"],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        in_shape=in_shape,
        out_shape=out_shape,
        inWidth=inWidth,
        outWidth=outWidth,
        preferred_impl_style=style,
        dataType=str(dt.name),
    )
    graph = helper.make_graph([node], "dwc", [inp], [outp])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="dwc"))
    model.set_tensor_datatype("inp", dt)
    model.set_tensor_datatype("outp", dt)
    return model


def measure(inWidth, outWidth, dt=DataType["INT8"], style="hls", mode="exact",
            pad_words=0):
    """Measure LUT/FF for a DWC. Word counts chosen so the *total* stream bits
    match the requested mode: 'exact' (NumIn*inW == NumOut*outW, no pad/crop),
    'pad' (out bits > in), 'crop' (out bits < in). Datapath LUTs are width-driven,
    so a few frames suffice. pad_words (legacy) still adds output words if set."""
    ibits = dt.bitwidth()
    in_els = inWidth // ibits
    out_els = outWidth // ibits
    # base word counts for one LCM-sized exact frame, repeated REPEAT times
    L = int(np.lcm(inWidth, outWidth))
    base_in = L // inWidth
    base_out = L // outWidth
    REPEAT = 2
    num_in = base_in * REPEAT
    num_out = base_out * REPEAT
    if mode == "pad" or pad_words:
        num_out += (pad_words or 1)     # extra output word(s) => zero padding
    elif mode == "crop":
        num_in += 1                     # extra input word(s) => cropping
    in_last = in_els * num_in
    out_last = out_els * num_out
    in_shape = [1, 1, in_last]
    out_shape = [1, 1, out_last]
    model = make_dwc(in_shape, out_shape, inWidth, outWidth, dt, style)
    model = model.transform(SpecializeLayers(TEST_PART))
    model = model.transform(GiveUniqueNodeNames())
    node = model.graph.node[0]
    impl = node.op_type
    if impl.endswith("_hls"):
        model = model.transform(PrepareIP(TEST_PART, 5))
        model = model.transform(HLSSynthIP())
        res = hls_synth_res_estimation(model)
        r = list(res.values())[0]
        return {"impl": impl, "LUT": int(r["LUT"]), "FF": int(r["FF"]),
                "BRAM": r.get("BRAM_18K", 0), "DSP": r.get("DSP48E", 0)}
    else:
        # RTL variant: use the node's own analytical estimate
        inst = getCustomOp(node)
        return {"impl": impl, "LUT": int(inst.lut_estimation()), "FF": -1,
                "BRAM": 0, "DSP": 0, "note": "rtl-analytical"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inw", type=int, default=None)
    ap.add_argument("--outw", type=int, default=None)
    ap.add_argument("--style", default="hls")
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--dt", default="INT8")
    args = ap.parse_args()
    dt = DataType[args.dt]
    if args.inw and args.outw:
        print(json.dumps(measure(args.inw, args.outw, dt, args.style, args.pad)))
        return
    # default sweep: downscaling multiple case, growing width
    print("in->out   impl                 LUT     FF")
    for inw, outw in [(32, 16), (64, 32), (128, 64), (256, 128),
                      (512, 256), (1024, 512), (48, 32), (96, 40)]:
        try:
            r = measure(inw, outw, dt, args.style, args.pad)
            print(f"{inw:>5}->{outw:<5} {r['impl']:<22} {r['LUT']:>6} {r.get('FF',-1):>6}")
        except Exception as e:
            print(f"{inw:>5}->{outw:<5} FAIL {type(e).__name__}: {str(e)[:80]}")


if __name__ == "__main__":
    main()
