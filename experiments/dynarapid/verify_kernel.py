"""Functional check of an island-built compute kernel (Alveo flow) at its AXI-Stream ports.

The island flow for Alveo (finn.util.rwislands.alveo) builds the compute partition only (AXI
streams, no IODMAs) and links it into the per-model v++ link. This script simulates, with the
same SystemVerilog testbench in xsim:

  isl: post-route functional netlist of the stitched core (core_dcp) inside the kernel wrapper
       (the stitched-IP interface ap_clk, ap_rst_n, s_axis_i, m_axis_j)
  ref: FINN's stitched IP of the same kernel model (RTL)

The input stream is random (optionally scaled per frame); the output streams must match.

Usage:
    python verify_kernel.py --kernel-dir <out>/rwislands/<kernel> --out <dir> [--frames 4]
"""

import argparse
import json
import numpy as np
import os
import re
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.dynarapid.graph import external_ports, kernel_wrapper_verilog
from finn.util.dynarapid.tools import run_vivado
from finn.util.rwislands.netlist import TOP_MODULE

from verify_accel import run_xsim, verilog_ports


def beats(node, inp=True):
    inst = getCustomOp(node)
    shape = inst.get_folded_input_shape() if inp else inst.get_folded_output_shape()
    return int(np.prod(shape[:-1]))


def tb_sv(win, wout, n_in, n_out, max_cycles):
    return "\n".join(
        [
            "`timescale 1ns/1ps",
            "module tb;",
            "  reg clk = 0; always #5 clk = ~clk;",
            "  reg rstn = 0;",
            "  reg [%d:0] mem [0:%d];" % (win - 1, n_in - 1),
            '  initial $readmemh("in.hex", mem);',
            "  integer i = 0, o = 0, cyc = 0, f;",
            "  wire [%d:0] din = mem[i < %d ? i : 0];" % (win - 1, n_in),
            "  wire vin = rstn && (i < %d);" % n_in,
            "  wire rin, vout;",
            "  wire [%d:0] dout;" % (wout - 1),
            "  dut u (.ap_clk(clk), .ap_rst_n(rstn), .s_axis_0_tdata(din), .s_axis_0_tvalid(vin),",
            "         .s_axis_0_tready(rin), .m_axis_0_tdata(dout), .m_axis_0_tvalid(vout),",
            "         .m_axis_0_tready(1'b1));",
            '  initial begin f = $fopen("out.hex", "w"); repeat (20) @(posedge clk); rstn = 1; end',
            "  always @(posedge clk) begin",
            "    cyc <= cyc + 1;",
            "    if (vin && rin) i <= i + 1;",
            '    if (rstn && vout) begin $fwrite(f, "%h\\n", dout); o <= o + 1; end',
            "    if (o == %d || cyc == %d) begin" % (n_out, max_cycles),
            '      $display("DONE outputs=%0d inputs=%0d cycles=%0d", o, i, cyc);',
            "      $fclose(f); $finish;",
            "    end",
            "  end",
            "endmodule",
            "",
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel-dir", required=True, help="out_dir of rw_islands_kernel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--max-cycles", type=int, default=5000000)
    ap.add_argument("--vary-amplitude", action="store_true", help="scale signed bytes per frame")
    ap.add_argument("--vary-unsigned", action="store_true", help="scale unsigned bytes per frame")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    km = ModelWrapper(os.path.join(args.kernel_dir, "kernel.onnx"))
    res = json.load(open(os.path.join(args.kernel_dir, "rwislands_kernel.json")))
    ins, outs = external_ports(km)
    assert len(ins) == 1 and len(outs) == 1, "single-stream kernels only"
    first = km.find_consumers(km.graph.input[0].name)[0]
    last = km.find_producer(km.graph.output[0].name)
    win, wout = ins[0]["width"], outs[0]["width"]
    n_in = beats(first, True) * args.frames
    n_out = beats(last, False) * args.frames
    np.random.seed(0)
    # random words, each byte random; optionally scaled per frame (signed bytes)
    nbytes = (win + 7) // 8
    data = np.random.randint(0, 256, size=(n_in, nbytes))
    if args.vary_amplitude:
        per = n_in // args.frames
        sig = data - 256 * (data >= 128)
        for fr in range(args.frames):
            sig[fr * per : (fr + 1) * per] = np.round(sig[fr * per : (fr + 1) * per] * (fr + 1) / args.frames)
        data = sig % 256
    if args.vary_unsigned:
        # e.g. image pixels (UINT8): frame f uses bytes scaled by (f + 1) / frames, no wraparound
        per = n_in // args.frames
        for fr in range(args.frames):
            data[fr * per : (fr + 1) * per] = (data[fr * per : (fr + 1) * per] * (fr + 1)) // args.frames
    mask = (1 << win) - 1
    words = []
    for row in data:
        v = 0
        for b in reversed(row):
            v = (v << 8) | int(b)
        words.append(v & mask)
    tb = tb_sv(win, wout, n_in, n_out, args.max_cycles)
    glbl = os.path.join(os.environ["XILINX_VIVADO"], "data", "verilog", "src", "glbl.v")
    results = {}
    for variant in ("isl", "ref"):
        sim = os.path.join(out, variant)
        os.makedirs(sim, exist_ok=True)
        open(os.path.join(sim, "tb.sv"), "w").write(tb)
        open(os.path.join(sim, "in.hex"), "w").write("\n".join("%x" % w for w in words) + "\n")
        if variant == "isl":
            netlist = os.path.join(sim, "core_funcsim.v")
            tcl = os.path.join(sim, "funcsim.tcl")
            with open(tcl, "w") as f:
                f.write("open_checkpoint %s\n" % res["core_dcp"])
                f.write("rename_ref -prefix_all isl_\n")
                f.write("write_verilog -mode funcsim -force -rename_top %s %s\n" % (TOP_MODULE, netlist))
            rc, _ = run_vivado(tcl, os.path.join(sim, "funcsim.log"), sim)
            assert rc == 0 and os.path.isfile(netlist), "netlist export failed"
            dut = os.path.join(sim, "dut.v")
            open(dut, "w").write(kernel_wrapper_verilog("dut", TOP_MODULE, ins, outs, black_box=False))
            log = run_xsim(sim, [netlist, dut, glbl, os.path.join(sim, "tb.sv")], libs=("unisims_ver", "secureip"))
        else:
            ref = km.transform(CreateStitchedIP(res["part"], res["clk_ns"]))
            proj = ref.get_metadata_prop("vivado_stitch_proj")
            wrapper_file = ref.get_metadata_prop("wrapper_filename")
            wname = os.path.basename(wrapper_file).rsplit(".", 1)[0]
            dut = os.path.join(sim, "dut.v")
            open(dut, "w").write(
                "module dut(input ap_clk, input ap_rst_n, input [%d:0] s_axis_0_tdata, input s_axis_0_tvalid,"
                " output s_axis_0_tready, output [%d:0] m_axis_0_tdata, output m_axis_0_tvalid,"
                " input m_axis_0_tready);\n  %s w (.ap_clk(ap_clk), .ap_rst_n(ap_rst_n),"
                " .s_axis_0_tdata(s_axis_0_tdata), .s_axis_0_tvalid(s_axis_0_tvalid),"
                " .s_axis_0_tready(s_axis_0_tready), .m_axis_0_tdata(m_axis_0_tdata),"
                " .m_axis_0_tvalid(m_axis_0_tvalid), .m_axis_0_tready(m_axis_0_tready));\nendmodule\n"
                % (win - 1, wout - 1, wname)
            )
            srcs = [l.strip() for l in open(os.path.join(proj, "all_verilog_srcs.txt")) if l.strip()]
            srcs = sorted(set(srcs) - {glbl})
            swg_pkg = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "swg", "swg_pkg.sv")
            if os.path.isfile(swg_pkg) and not any(f.endswith("swg_pkg.sv") for f in srcs):
                srcs.append(swg_pkg)
            srcs = [f for f in srcs if f.endswith("_pkg.sv")] + [f for f in srcs if not f.endswith("_pkg.sv")]
            log = run_xsim(sim, srcs + [glbl, dut, os.path.join(sim, "tb.sv")], libs=("unisims_ver",))
        m = re.search(r"DONE outputs=(\d+) inputs=(\d+) cycles=(\d+)", log)
        outf = os.path.join(sim, "out.hex")
        results[variant] = {
            "outputs": int(m.group(1)) if m else None,
            "cycles": int(m.group(3)) if m else None,
            "out": open(outf).read().split() if os.path.isfile(outf) else None,
        }
    same = results["isl"]["out"] is not None and results["isl"]["out"] == results["ref"]["out"]
    summary = {
        "frames": args.frames,
        "expected_outputs": n_out,
        "isl_outputs": results["isl"]["outputs"],
        "ref_outputs": results["ref"]["outputs"],
        "isl_cycles": results["isl"]["cycles"],
        "ref_cycles": results["ref"]["cycles"],
        "outputs_match": same and results["ref"]["outputs"] == n_out,
        "out_head_ref": (results["ref"]["out"] or [])[:8],
        "out_head_isl": (results["isl"]["out"] or [])[:8],
    }
    json.dump(summary, open(os.path.join(out, "verify_kernel.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
