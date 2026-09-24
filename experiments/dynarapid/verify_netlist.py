"""Functional check of a DynaRapid-routed kernel against FINN's stitched-IP RTL.

The routed checkpoint is exported as a post-route functional netlist (Vivado
write_verilog -mode funcsim), wrapped with the stitched-IP interface and simulated
with FINN's rtlsim (xsi) on random inputs; the outputs must match the rtlsim of
the regular stitched IP of the same dataflow model.

Usage:
    python verify_netlist.py --model <dir>/dataflow_ipgen.onnx --dcp <routed.dcp> --out <dir>
"""

import argparse
import json
import numpy as np
import os
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import gen_finn_dt_tensor

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.dynarapid.graph import external_ports, kernel_wrapper_verilog
from finn.util.dynarapid.tools import run_vivado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dcp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument(
        "--component",
        action="store_true",
        help="the checkpoint is a single component (adapter port names), e.g. its synth DCP",
    )
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    model = ModelWrapper(args.model)
    part = model.get_metadata_prop("dynarapid_fpga_part")
    clk_ns = float(model.get_metadata_prop("dynarapid_clk_ns"))

    # golden: stitched IP of the same model (no synthesis), simulated with xsi
    golden = model.transform(CreateStitchedIP(part, clk_ns))
    golden.set_metadata_prop("exec_mode", "rtlsim")

    # DynaRapid: post-route functional netlist of the routed checkpoint
    core = "dynarapid_core"
    netlist = os.path.join(out, core + ".v")
    tcl = os.path.join(out, "funcsim.tcl")
    with open(tcl, "w") as f:
        f.write("open_checkpoint %s\n" % args.dcp)
        f.write("rename_ref -prefix_all dr_\n")
        f.write("write_verilog -mode funcsim -force -rename_top %s %s\n" % (core, netlist))
    rc, _ = run_vivado(tcl, os.path.join(out, "funcsim.log"), out)
    assert rc == 0 and os.path.isfile(netlist), "netlist export failed, see funcsim.log"
    ins, outs = external_ports(model)
    if args.component:
        # single-node model: the checkpoint has the DynaRapid adapter ports
        ins = [
            dict(
                p, data="dataInArray_%d" % i, valid="pValidArray_%d" % i, ready="readyArray_%d" % i
            )
            for i, p in enumerate(ins)
        ]
        outs = [
            dict(
                p, data="dataOutArray_%d" % j, valid="validArray_%d" % j, ready="nReadyArray_%d" % j
            )
            for j, p in enumerate(outs)
        ]
    wrapper = "finn_dynarapid_wrapper"
    wrapper_file = os.path.join(out, wrapper + ".v")
    wtxt = kernel_wrapper_verilog(wrapper, core, ins, outs, black_box=False)
    # designs routed with DynaRapid's own clock buffer have a clkin port instead of clk
    with open(netlist) as f:
        head = f.read().split("module %s" % core, 1)[1].split(";", 1)[0]
    if "clkin" in head:
        wtxt = wtxt.replace(".clk(ap_clk)", ".clkin(ap_clk)")
    with open(wrapper_file, "w") as f:
        f.write(wtxt)
    with open(os.path.join(out, "all_verilog_srcs.txt"), "w") as f:
        f.write("%s\n%s\n" % (netlist, wrapper_file))
    dr = ModelWrapper(args.model)
    dr.set_metadata_prop("exec_mode", "rtlsim")
    dr.set_metadata_prop("wrapper_filename", wrapper_file)
    dr.set_metadata_prop("vivado_stitch_proj", out)
    dr.set_metadata_prop("vivado_stitch_ifnames", golden.get_metadata_prop("vivado_stitch_ifnames"))

    iname = model.graph.input[0].name
    oname = model.graph.output[0].name
    ishape = list(model.get_tensor_shape(iname))
    ishape[0] = 1
    idt = model.get_tensor_datatype(iname)
    res = {"frames": args.frames, "match": []}
    np.random.seed(0)
    for i in range(args.frames):
        x = gen_finn_dt_tensor(idt, ishape)
        y_ref = execute_onnx(golden, {iname: x})[oname]
        y_dr = execute_onnx(dr, {iname: x})[oname]
        ok = bool(np.array_equal(y_ref, y_dr))
        res["match"].append(ok)
        print(
            "frame %d: %s" % (i, "match" if ok else "MISMATCH"),
            y_ref.flatten()[:8],
            y_dr.flatten()[:8],
        )
    res["all_match"] = all(res["match"])
    with open(os.path.join(out, "verify.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
