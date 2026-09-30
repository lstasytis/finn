"""Functional check of a DynaRapid-routed accelerator (IODMAs included) at its AXI ports.

The accelerator of the DynaRapid bitfile flow (finn.util.dynarapid.zynq) is the part of the
design that DynaRapid places and routes: IODMAs + compute layers, with the IODMA AXI
channels packed into DynaRapid elastic channels. This script simulates

  dr : post-route functional netlist of the routed accelerator + the shell's bridge
       modules (the same Verilog the shell uses to unpack the channels into AXI)
  ref: FINN's regular stitched IP of the same accelerator graph (RTL)

in the same SystemVerilog testbench with xsim: an AXI-Lite master programs the IODMAs like
the FINN driver does (buffer addresses, numReps, ap_start), AXI memory models serve the
IODMA masters. The input buffer is random; the output buffers must be identical.

Usage:
    python verify_accel.py --accel-dir <out>/dynarapid --out <dir> [--frames 2]
"""

import argparse
import glob
import json
import numpy as np
import os
import re
import subprocess
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.dynarapid.graph import mm_ports
from finn.util.dynarapid.shell import CORE_MODULE, bridge_verilog
from finn.util.dynarapid.tools import run_vivado

IN_BASE = 0x10000000
OUT_BASE = 0x20000000


def verilog_ports(path, module):
    """{port: width} of a Verilog module (ANSI or non-ANSI declarations)."""
    txt = open(path).read()
    body = txt.split("module %s" % module, 1)[1].split("endmodule", 1)[0]
    ports = {}
    for d, rng, name in re.findall(
        r"\b(input|output)\s+(?:wire\s+|reg\s+)?(\[\s*\d+\s*:\s*\d+\s*\])?\s*(\w+)", body
    ):
        w = 1
        if rng:
            a, b = [int(x) for x in rng.strip("[]").split(":")]
            w = abs(a - b) + 1
        ports[name] = w
    return ports


DRAIN_CYCLES = 5000


def tb_sv(dmas, dw, nbytes_in, nbytes_out, frames, cycles):
    """Testbench: dmas = [{"k", "dir", "mem_dw"}]; the DUT is module `dut`."""
    L = ["`timescale 1ns/1ps", "module tb;", "  reg clk = 0; always #2.5 clk = ~clk;", "  reg rstn = 0;"]
    conns = ["    .ap_clk(clk)", "    .ap_rst_n(rstn)"]
    for d in dmas:
        k, mdw = d["k"], d["mem_dw"]
        c, m = "c%d_" % k, "m%d_" % k
        sig = [
            (c + "awaddr", 32), (c + "awvalid", 1), (c + "awready", 1), (c + "wdata", 32),
            (c + "wstrb", 4), (c + "wvalid", 1), (c + "wready", 1), (c + "bresp", 2),
            (c + "bvalid", 1), (c + "bready", 1), (c + "araddr", 32), (c + "arvalid", 1),
            (c + "arready", 1), (c + "rdata", 32), (c + "rresp", 2), (c + "rvalid", 1),
            (c + "rready", 1),
            (m + "awaddr", 64), (m + "awlen", 8), (m + "awvalid", 1), (m + "awready", 1),
            (m + "wdata", mdw), (m + "wstrb", mdw // 8), (m + "wlast", 1), (m + "wvalid", 1),
            (m + "wready", 1), (m + "bresp", 2), (m + "bvalid", 1), (m + "bready", 1),
            (m + "araddr", 64), (m + "arlen", 8), (m + "arvalid", 1), (m + "arready", 1),
            (m + "rdata", mdw), (m + "rresp", 2), (m + "rlast", 1), (m + "rvalid", 1),
            (m + "rready", 1),
        ]
        for s, w in sig:
            L.append("  wire %s%s;" % ("[%d:0] " % (w - 1) if w > 1 else "", s))
            conns.append("    .%s(%s)" % (s, s))
        # AXI-Lite master (driven by tasks)
        L += [
            "  reg [31:0] %sawaddr_r = 0, %swdata_r = 0, %saraddr_r = 0;" % (c, c, c),
            "  reg %sawvalid_r = 0, %swvalid_r = 0, %sarvalid_r = 0, %sbready_r = 0, %srready_r = 0;"
            % (c, c, c, c, c),
            "  assign %sawaddr = %sawaddr_r; assign %swdata = %swdata_r; assign %saraddr = %saraddr_r;"
            % (c, c, c, c, c, c),
            "  assign %sawvalid = %sawvalid_r; assign %swvalid = %swvalid_r; assign %sarvalid = %sarvalid_r;"
            % (c, c, c, c, c, c),
            "  assign %sbready = %sbready_r; assign %srready = %srready_r; assign %swstrb = 4'hf;"
            % (c, c, c, c, c),
            "  task lite_write_%d(input [31:0] a, input [31:0] v);" % k,
            "    begin",
            "      @(posedge clk); %sawaddr_r <= a; %sawvalid_r <= 1; %swdata_r <= v; %swvalid_r <= 1; %sbready_r <= 1;"
            % (c, c, c, c, c),
            "      fork",
            "        begin @(posedge clk); while (!%sawready) @(posedge clk); %sawvalid_r <= 0; end" % (c, c),
            "        begin @(posedge clk); while (!%swready) @(posedge clk); %swvalid_r <= 0; end" % (c, c),
            "      join",
            "      while (!%sbvalid) @(posedge clk);" % c,
            "      @(posedge clk); %sbready_r <= 0;" % c,
            "    end",
            "  endtask",
            "  task lite_read_%d(input [31:0] a, output [31:0] v);" % k,
            "    begin",
            "      @(posedge clk); %saraddr_r <= a; %sarvalid_r <= 1; %srready_r <= 1;" % (c, c, c),
            "      @(posedge clk); while (!%sarready) @(posedge clk); %sarvalid_r <= 0;" % (c, c),
            "      while (!%srvalid) @(posedge clk); v = %srdata;" % (c, c),
            "      @(posedge clk); %srready_r <= 0;" % c,
            "    end",
            "  endtask",
        ]
        # AXI memory slave (one outstanding burst per direction, INCR, full-width beats)
        nb = mdw // 8
        L += [
            "  reg [%d:0] mem%d [longint];" % (mdw - 1, k),
            "  reg %sarready_r = 0, %srvalid_r = 0, %srlast_r = 0; reg [%d:0] %srdata_r = 0;"
            % (m, m, m, mdw - 1, m),
            "  reg %sawready_r = 0, %swready_r = 0, %sbvalid_r = 0;" % (m, m, m),
            "  assign %sarready = %sarready_r; assign %srvalid = %srvalid_r; assign %srlast = %srlast_r;"
            % (m, m, m, m, m, m),
            "  assign %srdata = %srdata_r; assign %srresp = 0; assign %sbresp = 0;" % (m, m, m, m),
            "  assign %sawready = %sawready_r; assign %swready = %swready_r; assign %sbvalid = %sbvalid_r;"
            % (m, m, m, m, m, m),
            "  initial begin : rd%d" % k,
            "    longint a; int n;",
            "    forever begin",
            "      %sarready_r <= 1; @(posedge clk); while (!%sarvalid) @(posedge clk);" % (m, m),
            "      a = %saraddr / %d; n = %sarlen + 1; %sarready_r <= 0;" % (m, nb, m, m),
            "      for (int i = 0; i < n; i++) begin",
            "        %srdata_r <= mem%d.exists(a + i) ? mem%d[a + i] : 0; %srvalid_r <= 1; %srlast_r <= (i == n - 1);"
            % (m, k, k, m, m),
            "        @(posedge clk); while (!%srready) @(posedge clk);" % m,
            "      end",
            "      %srvalid_r <= 0; %srlast_r <= 0;" % (m, m),
            "    end",
            "  end",
            "  initial begin : wr%d" % k,
            "    longint a; int n;",
            "    forever begin",
            "      %sawready_r <= 1; @(posedge clk); while (!%sawvalid) @(posedge clk);" % (m, m),
            "      a = %sawaddr / %d; n = %sawlen + 1; %sawready_r <= 0; %swready_r <= 1;" % (m, nb, m, m, m),
            "      for (int i = 0; i < n; i++) begin",
            "        @(posedge clk); while (!%swvalid) @(posedge clk);" % m,
            "        for (int b = 0; b < %d; b++) if (%swstrb[b]) begin" % (nb, m),
            "          if (!mem%d.exists(a + i)) mem%d[a + i] = 0;" % (k, k),
            "          mem%d[a + i][b*8 +: 8] = %swdata[b*8 +: 8];" % (k, m),
            "        end",
            "      end",
            "      %swready_r <= 0; %sbvalid_r <= 1; @(posedge clk); while (!%sbready) @(posedge clk); %sbvalid_r <= 0;"
            % (m, m, m, m),
            "    end",
            "  end",
        ]
    L.append("  dut dut_i (\n%s\n  );" % ",\n".join(conns))
    idmas = [d for d in dmas if d["dir"] == "in"]
    odmas = [d for d in dmas if d["dir"] == "out"]
    L += [
        "  initial begin : main",
        "    reg [31:0] st; int fd; longint cyc; reg [7:0] b;",
    ]
    # input buffer: random bytes from in.hex (one byte per line)
    for d in idmas:
        k, nb = d["k"], d["mem_dw"] // 8
        L += [
            "    fd = $fopen(\"in%d.hex\", \"r\");" % k,
            "    for (int i = 0; i < %d; i++) begin" % nbytes_in,
            "      void'($fscanf(fd, \"%h\\n\", b));",
            "      if (!mem%d.exists(%d/%d + i/%d)) mem%d[%d/%d + i/%d] = 0;" % (k, IN_BASE, nb, nb, k, IN_BASE, nb, nb),
            "      mem%d[%d/%d + i/%d][(i%%%d)*8 +: 8] = b;" % (k, IN_BASE, nb, nb, nb),
            "    end",
            "    $fclose(fd);",
        ]
    L += ["    repeat (20) @(posedge clk); rstn <= 1; repeat (20) @(posedge clk);"]
    for d in odmas:
        k = d["k"]
        L += [
            "    lite_write_%d(32'h10, %d); lite_write_%d(32'h14, 0);" % (k, OUT_BASE, k),
            "    lite_write_%d(32'h1c, %d); lite_write_%d(32'h00, 1);" % (k, frames, k),
        ]
    for d in idmas:
        k = d["k"]
        L += [
            "    lite_write_%d(32'h10, %d); lite_write_%d(32'h14, 0);" % (k, IN_BASE, k),
            "    lite_write_%d(32'h1c, %d); lite_write_%d(32'h00, 1);" % (k, frames, k),
        ]
    k = odmas[0]["k"]
    nb = odmas[0]["mem_dw"] // 8
    L += [
        "    cyc = 0; st = 0;",
        "    while (!(st & 2) && cyc < %d) begin lite_read_%d(32'h00, st); cyc = cyc + 1; end" % (cycles, k),
        "    $display(\"DONE status=%h polls=%0d time=%0t\", st, cyc, $time);",
        # drain: with Vivado 2024.2 HLS the IODMA reports ap_done before its last write burst
        # reaches the memory model (seen in the FINN reference: last word missing at DONE)
        "    repeat (%d) @(posedge clk);" % DRAIN_CYCLES,
        "    fd = $fopen(\"out.hex\", \"w\");",
        "    for (int i = 0; i < %d; i++) begin" % nbytes_out,
        "      if (mem%d.exists(%d/%d + i/%d)) $fdisplay(fd, \"%%02h\", mem%d[%d/%d + i/%d][(i%%%d)*8 +: 8]);"
        % (k, OUT_BASE, nb, nb, k, OUT_BASE, nb, nb, nb),
        "      else $fdisplay(fd, \"xx\");",
        "    end",
        "    $fclose(fd);",
        "    $finish;",
        "  end",
        "endmodule",
    ]
    return "\n".join(L) + "\n"


def dut_dr(ports, core_ports, gm_dw):
    """dut: shell bridges + DynaRapid core netlist, mapped to the testbench signals."""
    L = ["module dut (input ap_clk, input ap_rst_n,"]
    decl, body = [], []
    for k, d in enumerate(ports):
        decl += _tb_port_decl(k, gm_dw[k])
        bp = ["    .ap_clk(ap_clk)", "    .ap_rst_n(ap_rst_n)"]
        for sig, tb in _axi_map(k, "m_axi_gmem_", "s_axi_control_"):
            bp.append("    .%s(%s)" % (sig, tb))
        for c in d["in"] + d["out"]:
            for p in (c["data"], c["valid"], c["ready"]):
                w = c["width"] if p == c["data"] else 1
                body.append("  wire %s%s;" % ("[%d:0] " % (w - 1) if w > 1 else "", p))
                bp.append("    .core_%s(%s)" % (p, p))
        bp.append("    .core_clk()")
        bp.append("    .core_rst()")
        body.append("  %s_bridge br%d (\n%s\n  );" % (d["id"], k, ",\n".join(bp)))
    cc = []
    for p in core_ports:
        if p == "clk":
            cc.append("    .clk(ap_clk)")
        elif p == "rst":
            cc.append("    .rst(ap_rst_n)")
        else:
            cc.append("    .%s(%s)" % (p, p))
    body.append("  %s core (\n%s\n  );" % (CORE_MODULE, ",\n".join(cc)))
    return "\n".join(L[:1]) + "\n" + ",\n".join(decl) + "\n);\n" + "\n".join(body) + "\nendmodule\n"


def dut_ref(wrapper, wports, n, gm_dw, gm_names, ctrl_names):
    """dut: FINN stitched IP wrapper mapped to the testbench signals."""
    decl, conns = [], ["    .ap_clk(ap_clk)", "    .ap_rst_n(ap_rst_n)"]
    for k in range(n):
        decl += _tb_port_decl(k, gm_dw[k])
        for sig, tb in _axi_map(k, gm_names[k] + "_", ctrl_names[k] + "_", lower=True):
            if sig in wports:
                conns.append("    .%s(%s)" % (sig, tb))
    # tie off the remaining inputs of the wrapper (AXI IDs, users, ...)
    return (
        "module dut (input ap_clk, input ap_rst_n,\n"
        + ",\n".join(decl)
        + "\n);\n  %s w (\n%s\n  );\nendmodule\n" % (wrapper, ",\n".join(conns))
    )


def _tb_port_decl(k, dw):
    c, m = "c%d_" % k, "m%d_" % k
    ins = [(c + "awaddr", 32), (c + "awvalid", 1), (c + "wdata", 32), (c + "wstrb", 4),
           (c + "wvalid", 1), (c + "bready", 1), (c + "araddr", 32), (c + "arvalid", 1),
           (c + "rready", 1), (m + "awready", 1), (m + "wready", 1), (m + "bresp", 2),
           (m + "bvalid", 1), (m + "arready", 1), (m + "rdata", dw), (m + "rresp", 2),
           (m + "rlast", 1), (m + "rvalid", 1)]
    outs = [(c + "awready", 1), (c + "wready", 1), (c + "bresp", 2), (c + "bvalid", 1),
            (c + "arready", 1), (c + "rdata", 32), (c + "rresp", 2), (c + "rvalid", 1),
            (m + "awaddr", 64), (m + "awlen", 8), (m + "awvalid", 1), (m + "wdata", dw),
            (m + "wstrb", dw // 8), (m + "wlast", 1), (m + "wvalid", 1), (m + "bready", 1),
            (m + "araddr", 64), (m + "arlen", 8), (m + "arvalid", 1), (m + "rready", 1)]
    r = []
    for s, w in ins:
        r.append("  input %s%s" % ("[%d:0] " % (w - 1) if w > 1 else "", s))
    for s, w in outs:
        r.append("  output %s%s" % ("[%d:0] " % (w - 1) if w > 1 else "", s))
    return r


def _axi_map(k, m_pre, c_pre, lower=False):
    """(DUT port, testbench signal) for the AXI signals the testbench uses."""
    c, m = "c%d_" % k, "m%d_" % k
    pairs = []
    for s in ("AWADDR", "AWVALID", "AWREADY", "WDATA", "WSTRB", "WVALID", "WREADY", "BRESP",
              "BVALID", "BREADY", "ARADDR", "ARVALID", "ARREADY", "RDATA", "RRESP", "RVALID",
              "RREADY"):
        pairs.append((c_pre + (s.lower() if lower else s), c + s.lower()))
    for s in ("AWADDR", "AWLEN", "AWVALID", "AWREADY", "WDATA", "WSTRB", "WLAST", "WVALID",
              "WREADY", "BRESP", "BVALID", "BREADY", "ARADDR", "ARLEN", "ARVALID", "ARREADY",
              "RDATA", "RRESP", "RLAST", "RVALID", "RREADY"):
        pairs.append((m_pre + (s.lower() if lower else s), m + s.lower()))
    return pairs


def run_xsim(sim_dir, srcs, top="tb", libs=()):
    """Compile and run with xvlog/xelab/xsim, return the simulation log."""
    sv = [s for s in srcs if s.endswith(".sv")]
    v = [s for s in srcs if s.endswith(".v")]
    vhd = [s for s in srcs if s.endswith((".vhd", ".vhdl"))]
    cmds = []
    if v:
        cmds.append(["xvlog", "--relax"] + v)
    if sv:
        cmds.append(["xvlog", "--sv", "--relax"] + sv)
    if vhd:
        cmds.append(["xvhdl", "--relax"] + vhd)
    lib_args = []
    for lib in libs:
        lib_args += ["-L", lib]
    tops = [top] + (["glbl"] if "unisims_ver" in libs else [])
    cmds.append(["xelab", "--relax", "-debug", "off", "-s", "snap"] + lib_args + tops)
    cmds.append(["xsim", "snap", "-R"])
    log = ""
    for c in cmds:
        p = subprocess.run(c, cwd=sim_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log += " ".join(c[:4]) + "\n" + p.stdout
        if p.returncode != 0:
            open(os.path.join(sim_dir, "sim.log"), "w").write(log)
            raise RuntimeError("%s failed, see %s" % (c[0], os.path.join(sim_dir, "sim.log")))
    open(os.path.join(sim_dir, "sim.log"), "w").write(log)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accel-dir", required=True, help="out_dir of dynarapid_zynq_build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--max-polls", type=int, default=200000)
    ap.add_argument("--variants", default="dr,ref", help="simulations to (re)run; others reused")
    ap.add_argument("--vary-amplitude", action="store_true", help="scale the input per frame")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    accel = ModelWrapper(os.path.join(args.accel_dir, "accel.onnx"))
    isl_json = os.path.join(args.accel_dir, "rwislands_zynq.json")
    if os.path.isfile(isl_json):
        # island flow (finn.util.rwislands): the RapidWright-stitched accelerator
        res_json = json.load(open(isl_json))
        routed = os.path.join(args.accel_dir, "work", "stitch", "accel_routed.dcp")
    else:
        res_json = json.load(open(os.path.join(args.accel_dir, "dynarapid_zynq.json")))
        routed = res_json["accel"]["routed_dcp"]
    part = res_json.get("part") or "xczu7ev-ffvc1156-2-e"
    clk_ns = res_json.get("clk_ns") or 5.0
    ports = mm_ports(accel)

    iodmas = [n for n in accel.graph.node if n.op_type.startswith("IODMA")]
    gm_dw = []
    for d in ports:
        w = [c for c in d["in"] + d["out"] if c["name"].endswith("_R") and c["name"].startswith("m_axi")][0]
        gm_dw.append([wd for s, wd in w["signals"] if s.endswith("RDATA")][0])
    # buffer sizes: input bytes of the first idma, output bytes of the first odma
    def nbytes(node, tensor):
        inst = getCustomOp(node)
        shape = accel.get_tensor_shape(tensor)
        n = int(np.prod(shape[1:]))
        bits = accel.get_tensor_datatype(tensor).bitwidth()
        # IODMAs transfer whole memory words per frame
        iw = inst.get_nodeattr("intfWidth")
        return ((n * bits + iw - 1) // iw) * iw // 8
    idma = [n for n in iodmas if getCustomOp(n).get_nodeattr("direction") == "in"][0]
    odma = [n for n in iodmas if getCustomOp(n).get_nodeattr("direction") == "out"][0]
    nin = nbytes(idma, idma.input[0]) * args.frames
    nout = nbytes(odma, odma.output[0]) * args.frames
    dmas = [
        {"k": k, "dir": "in" if d["id"].startswith("i") else "out", "mem_dw": gm_dw[k]}
        for k, d in enumerate(ports)
    ]
    tb = tb_sv(dmas, None, nin, nout, args.frames, args.max_polls)
    np.random.seed(0)
    data = np.random.randint(0, 256, size=nin)
    if args.vary_amplitude:
        # frame f: signed random bytes scaled by (f + 1) / frames, so that the frames exercise
        # different activation levels (uniform noise alone tends to give one class)
        per = nin // args.frames
        sig = data.astype(np.int64) - 256 * (data >= 128)
        for f in range(args.frames):
            sl = slice(f * per, (f + 1) * per)
            sig[sl] = np.round(sig[sl] * (f + 1) / args.frames)
        data = (sig % 256).astype(np.int64)

    results = {}
    for variant in ("dr", "ref"):
        sim = os.path.join(out, variant)
        os.makedirs(sim, exist_ok=True)
        if variant not in args.variants.split(","):
            lf = os.path.join(sim, "sim.log")
            log = open(lf, errors="ignore").read() if os.path.isfile(lf) else ""
            m = re.search(r"DONE status=(\w+) polls=(\d+) time=(\d+)", log)
            outf = os.path.join(sim, "out.hex")
            results[variant] = {
                "done": m is not None and (int(m.group(1), 16) & 2) != 0,
                "sim_ns": int(m.group(3)) / 1000.0 if m else None,
                "out": open(outf).read().split() if os.path.isfile(outf) else None,
            }
            continue
        open(os.path.join(sim, "tb.sv"), "w").write(tb)
        for d in dmas:
            if d["dir"] == "in":
                open(os.path.join(sim, "in%d.hex" % d["k"]), "w").write("\n".join("%02x" % b for b in data) + "\n")
        if variant == "dr":
            netlist = os.path.join(sim, "core_funcsim.v")
            tcl = os.path.join(sim, "funcsim.tcl")
            with open(tcl, "w") as f:
                f.write("open_checkpoint %s\n" % routed)
                f.write("rename_ref -prefix_all dr_\n")
                f.write("write_verilog -mode funcsim -force -rename_top %s %s\n" % (CORE_MODULE, netlist))
            rc, _ = run_vivado(tcl, os.path.join(sim, "funcsim.log"), sim)
            assert rc == 0 and os.path.isfile(netlist), "netlist export failed"
            core_ports = verilog_ports(netlist, CORE_MODULE)
            br = os.path.join(sim, "bridges.v")
            open(br, "w").write("".join(bridge_verilog(d) for d in ports))
            dut = os.path.join(sim, "dut.v")
            open(dut, "w").write(dut_dr(ports, core_ports, gm_dw))
            glbl = os.path.join(os.environ["XILINX_VIVADO"], "data", "verilog", "src", "glbl.v")
            srcs = [netlist, br, dut, glbl, os.path.join(sim, "tb.sv")]
            log = run_xsim(sim, srcs, libs=("unisims_ver", "secureip"))
        else:
            ref = accel.transform(CreateStitchedIP(part, clk_ns))
            proj = ref.get_metadata_prop("vivado_stitch_proj")
            wrapper_file = ref.get_metadata_prop("wrapper_filename")
            wname = os.path.basename(wrapper_file).rsplit(".", 1)[0]
            wports = verilog_ports(wrapper_file, wname)
            ifn = eval(ref.get_metadata_prop("vivado_stitch_ifnames"))
            gm_names = [n for n, _ in ifn["aximm"]]
            ctrl_names = list(ifn["axilite"])
            dut = os.path.join(sim, "dut.v")
            open(dut, "w").write(dut_ref(wname, wports, len(ports), gm_dw, gm_names, ctrl_names))
            srcs = [l.strip() for l in open(os.path.join(proj, "all_verilog_srcs.txt")) if l.strip()]
            glbl = os.path.join(os.environ["XILINX_VIVADO"], "data", "verilog", "src", "glbl.v")
            srcs = sorted(set(srcs) - {glbl})
            # SystemVerilog packages must be compiled before their users (sorting breaks the
            # order); the SWG package is not always listed
            swg_pkg = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "swg", "swg_pkg.sv")
            if os.path.isfile(swg_pkg) and not any(f.endswith("swg_pkg.sv") for f in srcs):
                srcs.append(swg_pkg)
            srcs = [f for f in srcs if f.endswith("_pkg.sv")] + [f for f in srcs if not f.endswith("_pkg.sv")]
            srcs += [glbl, dut, os.path.join(sim, "tb.sv")]
            log = run_xsim(sim, srcs, libs=("unisims_ver",))
        m = re.search(r"DONE status=(\w+) polls=(\d+) time=(\d+)", log)
        results[variant] = {
            "done": m is not None and (int(m.group(1), 16) & 2) != 0,
            "sim_ns": int(m.group(3)) / 1000.0 if m else None,
            "out": open(os.path.join(sim, "out.hex")).read().split() if os.path.isfile(os.path.join(sim, "out.hex")) else None,
        }
    same = results["dr"]["out"] is not None and results["dr"]["out"] == results["ref"]["out"]
    written = results["ref"]["out"] is not None and any(x != "xx" for x in results["ref"]["out"])
    summary = {
        "frames": args.frames,
        "dr_done": results["dr"]["done"],
        "ref_done": results["ref"]["done"],
        "dr_sim_ns": results["dr"]["sim_ns"],
        "ref_sim_ns": results["ref"]["sim_ns"],
        "output_bytes": nout,
        "ref_wrote_output": written,
        "outputs_match": same,
        "out_head_ref": results["ref"]["out"][:16] if results["ref"]["out"] else None,
        "out_head_dr": results["dr"]["out"][:16] if results["dr"]["out"] else None,
    }
    json.dump(summary, open(os.path.join(out, "verify_accel.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()