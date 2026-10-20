# Issue 011: take a build's ipgen'd dataflow model, override some FIFO depths,
# regenerate those FIFOs, stitch and run stitched-IP rtlsim on the golden input.
# Usage: fifo_override.py <build_dir> <golden_prefix> NAME=DEPTH ...  (no overrides = rerun)
import os
import sys

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP

part, clk = "xczu7ev-ffvc1156-2-e", 10.0
bdir, gold = sys.argv[1], sys.argv[2]
model = ModelWrapper(bdir + "/intermediate_models/step_hw_ipgen.onnx")
for arg in sys.argv[3:]:
    name, depth = arg.split("=")
    inst = getCustomOp(model.get_node_from_name(name))
    print(name, "depth", inst.get_nodeattr("depth"), "->", depth)
    inst.set_nodeattr("depth", int(depth))
    for a in ["code_gen_dir_ipgen", "ipgen_path", "ip_path"]:
        inst.set_nodeattr(a, "")
model = model.transform(PrepareIP(part, clk)).transform(HLSSynthIP())
model = model.transform(CreateStitchedIP(part, clk))
# diagnostic only: MVUFIX=<patched mvu_vvu_axi.sv> swaps it into the rtlsim sources
if os.environ.get("MVUFIX"):
    srcs = model.get_metadata_prop("vivado_stitch_proj") + "/all_verilog_srcs.txt"
    lst = [x if not x.endswith("/mvu_vvu_axi.sv") else os.environ["MVUFIX"] for x in open(srcs).read().split()]
    open(srcs, "w").write("\n".join(lst))
    print("rtlsim with", os.environ["MVUFIX"])
model.set_metadata_prop("exec_mode", "rtlsim")
model.save(os.environ["FINN_BUILD_DIR"] + "/ovr_model_%d.onnx" % os.getpid())
inp = np.load(gold + "_input.npy")
golden = np.load(gold + "_output.npy")
iname, oname = model.graph.input[0].name, model.graph.output[0].name
ishape = tuple(model.get_tensor_shape(iname))
if os.environ.get("NCHW2NHWC"):
    inp = inp.transpose(0, 2, 3, 1)  # NCHW golden, NHWC child input (the parent transposes)
out = oxe.execute_onnx(model, {iname: inp})[oname]
print("cycles", model.get_metadata_prop("cycles_rtlsim"))
print("out", out.flatten())
np.save(os.environ["FINN_BUILD_DIR"] + "/ovr_out_%d.npy" % os.getpid(), out)
print("maxdiff vs golden", np.abs(out.flatten() - golden.flatten()).max(), flush=True)
