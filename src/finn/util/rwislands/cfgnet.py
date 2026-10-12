# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration network of the tile overlay: the AXI-Lite slaves of compute nodes (runtime-
writeable weights, e.g. ResNet50's 50 URAM MVAUs) in any tile, reached through one request and
one response stream along the snake (the control network of PLD / PRflow, here a chain).

Without it every node with a shell port had to sit in tile 0: ResNet50 (50 AXI-Lite MVAUs) could
not be planned, and carrying the 50 AXI-Lite ports over the links as streams would need ~3450
forward / ~2050 backward bits at the first tile edge (the bundles have 2304 / 32-128).

    shell AXI-Lite ports (one per node) -- tile 0: funnel -- req -->  tile 1: node -- req --> ...
                                                         <-- resp --          <-- resp --
  * funnel (tile 0): takes one transaction (AW + W, or AR) from one of its slave ports at a time,
    sends {we, tag, addr, wdata, wstrb} as a request packet, waits for the response packet
    {resp, rdata} and returns it on the port's B / R channel. One transaction in flight in the
    whole network: no ordering or arbitration issues anywhere (configuration only: ResNet50's
    weights are ~1 M AXI-Lite writes, ~100 cycles each over a 20-tile chain: ~0.3 s).
  * chain node (tile k): a request for one of its local tags is executed on that node's AXI-Lite
    port, any other one is passed east; responses from the east and the local one go west. The
    last node answers unknown tags with SLVERR.
  * between tiles: the request (forward) and response (backward) streams use the normal link
    transmitters / receivers on reserved bundle bits (forward [0, RQ] request data + valid, RQ + 1
    response ready; backward [0] request ready, [1, 35] response data + valid).
"""

import hashlib

RESP_W = 34  # {resp[1:0], rdata[31:0]} (the R channel's packing: RDATA LSBs, RRESP above)
W_W = 36  # {wstrb[3:0], wdata[31:0]}


def req_width(addr_w, n_tags):
    return 1 + tag_width(n_tags) + addr_w + W_W


def tag_width(n_tags):
    return max(1, (max(1, n_tags) - 1).bit_length())


def reserved_bits(addr_w, n_tags):
    """(forward, backward) bundle bits the configuration chain takes on each tile edge it uses."""
    return req_width(addr_w, n_tags) + 2, RESP_W + 2


def _clog2(n):
    return max(1, (max(1, n) - 1).bit_length())


def funnel_name(n, addr_w, n_tags):
    return "rwi_cfg_funnel_n%d_a%d_t%d" % (n, addr_w, tag_width(n_tags))


def node_name(n_local, addr_w, n_tags, east):
    return "rwi_cfg_node_l%d_a%d_t%d_e%d" % (n_local, addr_w, tag_width(n_tags), int(bool(east)))


def parse_name(name):
    """("funnel" | "node", {"n" | "l", "a", "t", "e"}) of a module name, or None."""
    import re

    m = re.match(r"rwi_cfg_funnel_n(\d+)_a(\d+)_t(\d+)$", name)
    if m:
        return "funnel", {"n": int(m.group(1)), "a": int(m.group(2)), "t": int(m.group(3))}
    m = re.match(r"rwi_cfg_node_l(\d+)_a(\d+)_t(\d+)_e(\d)$", name)
    if m:
        return "node", {"l": int(m.group(1)), "a": int(m.group(2)), "t": int(m.group(3)), "e": int(m.group(4))}
    return None


def module_verilog(name):
    kind, p = parse_name(name)
    n_tags = 1 << p["t"]
    if kind == "funnel":
        return funnel_verilog(p["n"], p["a"], n_tags)[1]
    return node_verilog(p["l"], p["a"], n_tags, p["e"])[1]


def module_stub(name):
    kind, p = parse_name(name)
    n_tags = 1 << p["t"]
    ports = funnel_ports(p["n"], p["a"], n_tags) if kind == "funnel" else node_ports(p["l"], p["a"], n_tags, p["e"])
    return stub(name, ports)


def funnel_ports(n, addr_w, n_tags):
    rq = req_width(addr_w, n_tags)
    p = ["input clk", "input rst"]
    for i, aw in enumerate([addr_w] * n):
        p += ["input [%d:0] s%d_aw_data" % (aw - 1, i), "input s%d_aw_valid" % i, "output s%d_aw_ready" % i,
              "input [%d:0] s%d_w_data" % (W_W - 1, i), "input s%d_w_valid" % i, "output s%d_w_ready" % i,
              "input [%d:0] s%d_ar_data" % (aw - 1, i), "input s%d_ar_valid" % i, "output s%d_ar_ready" % i,
              "output [%d:0] s%d_r_data" % (RESP_W - 1, i), "output s%d_r_valid" % i, "input s%d_r_ready" % i,
              "output [1:0] s%d_b_data" % i, "output s%d_b_valid" % i, "input s%d_b_ready" % i]
    p += ["output [%d:0] q_data" % (rq - 1), "output q_valid", "input q_ready",
          "input [%d:0] p_data" % (RESP_W - 1), "input p_valid", "output p_ready"]
    return p


def funnel_verilog(n, addr_w, n_tags):
    """Funnel of n AXI-Lite slave ports (AW / AR payloads addr_w bits, narrower ones zero-extended
    by the caller; tag = port index)."""
    t, a = tag_width(n_tags), addr_w
    aws = [a] * n
    rq = req_width(a, n_tags)
    name = funnel_name(n, a, n_tags)
    L = ["module %s (" % name, ",\n".join("    " + x for x in funnel_ports(n, a, n_tags)), ");"]
    L += ["    localparam IDLE = 2'd0, SEND = 2'd1, WAIT = 2'd2, RESP = 2'd3;",
          "    reg [1:0] st = IDLE;",
          "    reg [%d:0] sel = 0;" % (_clog2(n) - 1),
          "    reg we = 1'b0;",
          "    reg [%d:0] addr = 0;" % (a - 1),
          "    reg [%d:0] wd = 0;" % (W_W - 1),
          "    reg [%d:0] rsp = 0;" % (RESP_W - 1),
          "    wire [%d:0] wreq, rreq;" % (n - 1),
          "    wire [%d:0] awd [0:%d];" % (a - 1, n - 1),
          "    wire [%d:0] ard [0:%d];" % (a - 1, n - 1),
          "    wire [%d:0] wdd [0:%d];" % (W_W - 1, n - 1)]
    for i, aw in enumerate(aws):
        pad = "" if aw == a else "{%d'b0}, " % (a - aw)
        L += ["    assign wreq[%d] = s%d_aw_valid & s%d_w_valid;" % (i, i, i),
              "    assign rreq[%d] = s%d_ar_valid;" % (i, i),
              "    assign awd[%d] = {%ss%d_aw_data};" % (i, pad, i),
              "    assign ard[%d] = {%ss%d_ar_data};" % (i, pad, i),
              "    assign wdd[%d] = s%d_w_data;" % (i, i)]
    # lowest port with a complete write or a read
    L += ["    reg [%d:0] first;" % (_clog2(n) - 1),
          "    reg any;",
          "    integer i;",
          "    always @* begin",
          "        any = 1'b0;",
          "        first = 0;",
          "        for (i = %d; i >= 0; i = i - 1)" % (n - 1),
          "            if (wreq[i] | rreq[i]) begin any = 1'b1; first = i; end",
          "    end",
          "    wire take = (st == IDLE) & any;",
          "    wire take_w = take & wreq[first];"]
    for i in range(n):
        L += ["    assign s%d_aw_ready = take_w & (first == %d);" % (i, i),
              "    assign s%d_w_ready = take_w & (first == %d);" % (i, i),
              "    assign s%d_ar_ready = take & ~wreq[first] & (first == %d);" % (i, i),
              "    assign s%d_b_valid = (st == RESP) & we & (sel == %d);" % (i, i),
              "    assign s%d_b_data = rsp[%d:%d];" % (i, RESP_W - 1, RESP_W - 2),
              "    assign s%d_r_valid = (st == RESP) & ~we & (sel == %d);" % (i, i),
              "    assign s%d_r_data = rsp;" % i]
    L += ["    reg resp_taken;",
          "    always @* begin",
          "        resp_taken = 1'b0;"]
    for i in range(n):
        L += ["        if (sel == %d) resp_taken = we ? s%d_b_ready : s%d_r_ready;" % (i, i, i)]
    L += ["    end",
          "    assign q_valid = (st == SEND);",
          "    assign q_data = {we, %s, addr, wd};" % ("{%d'b0, sel}" % (t - _clog2(n)) if t > _clog2(n) else "sel[%d:0]" % (t - 1)),
          "    assign p_ready = (st == WAIT);",
          "    always @(posedge clk) begin",
          "        if (!rst) begin",
          "            st <= IDLE;",
          "        end else begin",
          "            case (st)",
          "                IDLE: if (any) begin",
          "                    sel <= first; we <= wreq[first];",
          "                    addr <= wreq[first] ? awd[first] : ard[first]; wd <= wdd[first];",
          "                    st <= SEND;",
          "                end",
          "                SEND: if (q_ready) st <= WAIT;",
          "                WAIT: if (p_valid) begin rsp <= p_data; st <= RESP; end",
          "                RESP: if (resp_taken) st <= IDLE;",
          "            endcase",
          "        end",
          "    end",
          "endmodule"]
    assert rq == 1 + t + a + W_W
    return name, "\n".join(L) + "\n"


def node_ports(n_local, addr_w, n_tags, east):
    rq = req_width(addr_w, n_tags)
    t = tag_width(n_tags)
    p = ["input clk", "input rst"] + (["input [%d:0] tags" % (n_local * t - 1)] if n_local else []) + [
         "input [%d:0] qi_data" % (rq - 1), "input qi_valid", "output qi_ready",
         "output [%d:0] po_data" % (RESP_W - 1), "output po_valid", "input po_ready"]
    if east:
        p += ["output [%d:0] qo_data" % (rq - 1), "output qo_valid", "input qo_ready",
              "input [%d:0] pi_data" % (RESP_W - 1), "input pi_valid", "output pi_ready"]
    for j, aw in enumerate([addr_w] * n_local):
        p += ["output [%d:0] m%d_aw_data" % (aw - 1, j), "output m%d_aw_valid" % j, "input m%d_aw_ready" % j,
              "output [%d:0] m%d_w_data" % (W_W - 1, j), "output m%d_w_valid" % j, "input m%d_w_ready" % j,
              "output [%d:0] m%d_ar_data" % (aw - 1, j), "output m%d_ar_valid" % j, "input m%d_ar_ready" % j,
              "input [%d:0] m%d_r_data" % (RESP_W - 1, j), "input m%d_r_valid" % j, "output m%d_r_ready" % j,
              "input [1:0] m%d_b_data" % j, "input m%d_b_valid" % j, "output m%d_b_ready" % j]
    return p


def node_verilog(n_local, addr_w, n_tags, east):
    """Chain node with n_local AXI-Lite masters; their tags come in on `tags` (constants in the tile
    netlist: one module per (n_local, widths, east), cached across models); AW / AR payloads
    addr_w bits (the caller connects the low bits of narrower ports)."""
    t, a = tag_width(n_tags), addr_w
    rq = req_width(a, n_tags)
    name = node_name(n_local, a, n_tags, east)
    nl = n_local
    local = [(None, a)] * nl
    L = ["module %s (" % name, ",\n".join("    " + x for x in node_ports(nl, a, n_tags, east)), ");"]
    L += ["    localparam IDLE = 3'd0, AWW = 3'd1, BW = 3'd2, ARR = 3'd3, RW = 3'd4, RESP = 3'd5;",
          "    reg [2:0] st = IDLE;",
          "    reg we = 1'b0, aw_done = 1'b0, w_done = 1'b0;",
          "    reg [%d:0] addr = 0;" % (a - 1),
          "    reg [%d:0] wd = 0;" % (W_W - 1),
          "    reg [%d:0] rsp = 0;" % (RESP_W - 1),
          "    reg [%d:0] cur = 0;" % (_clog2(max(nl, 1)) - 1),
          "    wire [%d:0] tag = qi_data[%d:%d];" % (t - 1, W_W + a + t - 1, W_W + a),
          "    wire hit;",
          "    wire [%d:0] idx;" % (_clog2(max(nl, 1)) - 1)]
    # (continuous assigns: an always @* without inputs never runs in simulation)
    if local:
        L += ["    assign hit = %s;" % " | ".join("(tag == tags[%d:%d])" % ((j + 1) * t - 1, j * t) for j in range(nl)),
              "    assign idx = %s0;" % "".join("(tag == tags[%d:%d]) ? %d : " % ((j + 1) * t - 1, j * t, j)
                                             for j in range(nl))]
    else:
        L += ["    assign hit = 1'b0;", "    assign idx = 0;"]
    L += [
          # a request this node executes: its own, or (last node) an unknown tag -> SLVERR
          "    wire mine = hit%s;" % ("" if east else " | 1'b1"),
          "    assign qi_ready = (st == IDLE) & mine%s;" % (" | ~mine & qo_ready" if east else "")]
    if east:
        L += ["    assign qo_data = qi_data;", "    assign qo_valid = qi_valid & ~mine;"]
    L += ["    wire m_aw_ready, m_w_ready, m_ar_ready, m_r_valid, m_b_valid;",
          "    wire [%d:0] m_r_data;" % (RESP_W - 1),
          "    wire [1:0] m_b_data;"]
    for sig, w in (("aw_ready", 1), ("w_ready", 1), ("ar_ready", 1), ("r_valid", 1), ("b_valid", 1),
                   ("r_data", RESP_W), ("b_data", 2)):
        L += ["    assign m_%s = %s%d'b0;" % (sig, "".join("(cur == %d) ? m%d_%s : " % (j, j, sig) for j in range(nl)), w)]
    for j, (_, aw) in enumerate(local):
        L += ["    assign m%d_aw_valid = (st == AWW) & ~aw_done & (cur == %d);" % (j, j),
              "    assign m%d_w_valid = (st == AWW) & ~w_done & (cur == %d);" % (j, j),
              "    assign m%d_ar_valid = (st == ARR) & (cur == %d);" % (j, j),
              "    assign m%d_b_ready = (st == BW) & (cur == %d);" % (j, j),
              "    assign m%d_r_ready = (st == RW) & (cur == %d);" % (j, j),
              "    assign m%d_aw_data = addr[%d:0];" % (j, aw - 1),
              "    assign m%d_ar_data = addr[%d:0];" % (j, aw - 1),
              "    assign m%d_w_data = wd;" % j]
    # responses: the local one, else the east's (one transaction in flight: never both)
    L += ["    assign po_valid = (st == RESP)%s;" % (" | pi_valid" if east else ""),
          "    assign po_data = (st == RESP) ? rsp : %s;" % ("pi_data" if east else "%d'b0" % RESP_W)]
    if east:
        L += ["    assign pi_ready = (st != RESP) & po_ready;"]
    L += ["    always @(posedge clk) begin",
          "        if (!rst) begin",
          "            st <= IDLE;",
          "        end else begin",
          "            case (st)",
          "                IDLE: if (qi_valid & mine) begin",
          "                    we <= qi_data[%d]; addr <= qi_data[%d:%d]; wd <= qi_data[%d:0]; cur <= idx;"
          % (rq - 1, W_W + a - 1, W_W, W_W - 1),
          "                    aw_done <= 1'b0; w_done <= 1'b0;",
          "                    if (!hit) begin rsp <= {2'b10, 32'b0}; st <= RESP; end",
          "                    else st <= qi_data[%d] ? AWW : ARR;" % (rq - 1),
          "                end",
          "                AWW: begin",
          "                    if (m_aw_ready) aw_done <= 1'b1;",
          "                    if (m_w_ready) w_done <= 1'b1;",
          "                    if ((aw_done | m_aw_ready) & (w_done | m_w_ready)) st <= BW;",
          "                end",
          "                BW: if (m_b_valid) begin rsp <= {m_b_data, 32'b0}; st <= RESP; end",
          "                ARR: if (m_ar_ready) st <= RW;",
          "                RW: if (m_r_valid) begin rsp <= m_r_data; st <= RESP; end",
          "                RESP: if (po_ready) st <= IDLE;",
          "            endcase",
          "        end",
          "    end",
          "endmodule"]
    return name, "\n".join(L) + "\n"


def stub(name, ports):
    return "(* black_box *)\nmodule %s (%s);\nendmodule\n" % (name, ", ".join(ports))


def testbench(n_tiles=3, per_tile=(2, 0, 3), aws=(15, 18, 12, 24, 16), pipe=2, n_ops=400, seed=1):
    """xsim test: funnel (tile 0) + chain nodes over link tx / anchors / rx, register-file AXI-Lite
    slaves (random ready / response delays) per tile; concurrent random writes and reads from all
    master ports; every read must return the last value written to that slave's register."""
    from finn.util.rwislands.links import ANCHOR, LINK_RX, LINK_TX, link_regs, rx_depth

    po, pi, ri, ro, _ = link_regs(pipe)
    a = max(aws)
    n = len(aws)
    assert sum(per_tile) == n
    rq = req_width(a, n)
    d = rx_depth(1, pipe)
    V = [LINK_TX, LINK_RX, ANCHOR]
    fname, fv = funnel_verilog(n, a, n)
    V.append(fv)
    tags = list(range(n))
    groups, k0 = [], 0
    for c in per_tile:
        groups.append(tags[k0:k0 + c])
        k0 += c
    names, seen = [], set()
    for k in range(1, n_tiles):
        loc = [(tg, aws[tg]) for tg in groups[k - 1]] if k - 1 < len(groups) else []
        nn, nv = node_verilog(len(loc), a, n, k < n_tiles - 1)
        if nn not in seen:
            V.append(nv)
            seen.add(nn)
        names.append((nn, loc))
    # register-file slave: 16 registers, random ready and response delays
    V.append("""module rf_slave #(parameter AW = 12) (input clk, input rst,
    input [AW-1:0] aw_data, input aw_valid, output reg aw_ready = 0,
    input [35:0] w_data, input w_valid, output reg w_ready = 0,
    input [AW-1:0] ar_data, input ar_valid, output reg ar_ready = 0,
    output reg [33:0] r_data = 0, output reg r_valid = 0, input r_ready,
    output reg [1:0] b_data = 0, output reg b_valid = 0, input b_ready);
    reg [31:0] regs [0:15];
    integer i; initial for (i = 0; i < 16; i = i + 1) regs[i] = 0;
    reg [AW-1:0] wa; reg [35:0] wdat; reg ha = 0, hw = 0;
    integer seed = AW;
    always @(posedge clk) begin
        aw_ready <= ~ha & ($random(seed) % 3 != 0);
        w_ready <= ~hw & ($random(seed) % 3 != 0);
        ar_ready <= ~r_valid & ($random(seed) % 3 != 0) & ~ar_ready;
        if (aw_valid & aw_ready) begin ha <= 1; wa <= aw_data; aw_ready <= 0; end
        if (w_valid & w_ready) begin hw <= 1; wdat <= w_data; w_ready <= 0; end
        if (ha & hw & ~b_valid & ($random(seed) % 2)) begin
            regs[wa[5:2]] <= wdat[31:0]; b_valid <= 1; b_data <= 0; ha <= 0; hw <= 0;
        end
        if (b_valid & b_ready) b_valid <= 0;
        if (ar_valid & ar_ready) begin r_valid <= 1; r_data <= {2'b00, regs[ar_data[5:2]]}; ar_ready <= 0; end
        if (r_valid & r_ready) r_valid <= 0;
    end
endmodule
""")
    T = ["`timescale 1ns/1ps", "module tb;", "    reg clk = 0; always #1.5 clk = ~clk;", "    reg rst = 0;"]
    # funnel slave ports <- testbench masters
    fconn = [".clk(clk)", ".rst(rst)"]
    for i, aw in enumerate(aws):
        for s, w in (("aw_data", aw), ("aw_valid", 1), ("aw_ready", 1), ("w_data", W_W), ("w_valid", 1), ("w_ready", 1),
                     ("ar_data", aw), ("ar_valid", 1), ("ar_ready", 1), ("r_data", RESP_W), ("r_valid", 1),
                     ("r_ready", 1), ("b_data", 2), ("b_valid", 1), ("b_ready", 1)):
            reg = s in ("aw_data", "aw_valid", "w_data", "w_valid", "ar_data", "ar_valid", "r_ready", "b_ready")
            T.append("    %s %ss%d_%s%s;" % ("reg" if reg else "wire", "[%d:0] " % (w - 1) if w > 1 else "", i, s,
                                            " = 0" if reg else ""))
            if s in ("aw_data", "ar_data") and aw < a:
                fconn.append(".s%d_%s({%d'b0, s%d_%s})" % (i, s, a - aw, i, s))
            else:
                fconn.append(".s%d_%s(s%d_%s)" % (i, s, i, s))
    # links between tiles k-1 and k: request q (forward), response p (backward)
    for k in range(1, n_tiles):
        T += ["    wire [%d:0] q%d_s_data, q%d_l_data, q%d_a_data, q%d_m_data;" % (rq - 1, k, k, k, k),
              "    wire q%d_s_valid, q%d_s_ready, q%d_l_valid, q%d_l_ready, q%d_a_valid, q%d_a_ready, q%d_m_valid, q%d_m_ready;"
              % ((k,) * 8),
              "    wire [%d:0] p%d_s_data, p%d_l_data, p%d_a_data, p%d_m_data;" % (RESP_W - 1, k, k, k, k),
              "    wire p%d_s_valid, p%d_s_ready, p%d_l_valid, p%d_l_ready, p%d_a_valid, p%d_a_ready, p%d_m_valid, p%d_m_ready;"
              % ((k,) * 8)]
        for x, w in (("q", rq), ("p", RESP_W)):
            T += ["    rwi_link_tx #(.W(%d), .PO(%d), .PI(%d)) %stx%d (.clk(clk), .rst(rst), .s_data(%s%d_s_data), "
                  ".s_valid(%s%d_s_valid), .s_ready(%s%d_s_ready), .l_data(%s%d_l_data), .l_valid(%s%d_l_valid), "
                  ".l_ready(%s%d_a_ready));" % ((w, po, pi, x, k) + (x, k) * 6),
                  "    rwi_anchor #(.W(%d)) %saf%d (.clk(clk), .d({%s%d_l_valid, %s%d_l_data}), .q({%s%d_a_valid, %s%d_a_data}));"
                  % (w + 1, x, k, x, k, x, k, x, k, x, k),
                  "    rwi_anchor #(.W(1)) %sab%d (.clk(clk), .d(%s%d_l_ready), .q(%s%d_a_ready));" % (x, k, x, k, x, k),
                  "    rwi_link_rx #(.W(%d), .DEPTH(%d), .SLACK(%d), .RI(%d), .RO(%d)) %srx%d (.clk(clk), .rst(rst), "
                  ".l_data(%s%d_a_data), .l_valid(%s%d_a_valid), .l_ready(%s%d_l_ready), .m_data(%s%d_m_data), "
                  ".m_valid(%s%d_m_valid), .m_ready(%s%d_m_ready));" % ((w, d, d - 3, ri, ro, x, k) + (x, k) * 6)]
    fconn += [".q_data(q1_s_data)", ".q_valid(q1_s_valid)", ".q_ready(q1_s_ready)",
              ".p_data(p1_m_data)", ".p_valid(p1_m_valid)", ".p_ready(p1_m_ready)"]
    T.append("    %s fun (%s);" % (fname, ", ".join(fconn)))
    for k in range(1, n_tiles):
        nn, loc = names[k - 1]
        c = [".clk(clk)", ".rst(rst)", ".qi_data(q%d_m_data)" % k, ".qi_valid(q%d_m_valid)" % k,
             ".qi_ready(q%d_m_ready)" % k, ".po_data(p%d_s_data)" % k, ".po_valid(p%d_s_valid)" % k,
             ".po_ready(p%d_s_ready)" % k]
        if k < n_tiles - 1:
            c += [".qo_data(q%d_s_data)" % (k + 1), ".qo_valid(q%d_s_valid)" % (k + 1), ".qo_ready(q%d_s_ready)" % (k + 1),
                  ".pi_data(p%d_m_data)" % (k + 1), ".pi_valid(p%d_m_valid)" % (k + 1), ".pi_ready(p%d_m_ready)" % (k + 1)]
        if loc:
            c.append(".tags({%s})" % ", ".join("%d'd%d" % (tag_width(n), tg) for tg, _ in reversed(loc)))
        for j, (tg, aw) in enumerate(loc):
            sl = []
            for s, w in (("aw_data", aw), ("aw_valid", 1), ("aw_ready", 1), ("w_data", W_W), ("w_valid", 1),
                         ("w_ready", 1), ("ar_data", aw), ("ar_valid", 1), ("ar_ready", 1), ("r_data", RESP_W),
                         ("r_valid", 1), ("r_ready", 1), ("b_data", 2), ("b_valid", 1), ("b_ready", 1)):
                wa = a if s in ("aw_data", "ar_data") else w
                T.append("    wire %sx%d_%s;" % ("[%d:0] " % (wa - 1) if wa > 1 else "", tg, s))
                c.append(".m%d_%s(x%d_%s)" % (j, s, tg, s))
                sl.append(".%s(x%d_%s%s)" % (s, tg, s, "[%d:0]" % (w - 1) if wa != w else ""))
            T.append("    rf_slave #(.AW(%d)) slv%d (.clk(clk), .rst(rst), %s);" % (aw, tg, ", ".join(sl)))
        T.append("    %s nd%d (%s);" % (nn, k, ", ".join(c)))
    # masters: per port a process doing random writes / reads, checking reads against a model
    T += ["    integer errors = 0, done = 0, ops = 0;"]
    for i in range(n):
        T += ["    initial begin : m%d" % i,
              "        reg [31:0] model [0:15]; integer j, r, s; reg [31:0] v;",
              "        s = %d;" % (seed + i),
              "        for (j = 0; j < 16; j = j + 1) model[j] = 0;",
              "        wait (rst);",
              "        repeat (%d) begin" % (n_ops // n),
              "            r = $random(s) & 15;",
              "            if ($random(s) & 1) begin",
              "                v = $random(s);",
              "                @(posedge clk); s%d_aw_data <= r << 2; s%d_w_data <= {4'hf, v}; s%d_aw_valid <= 1; s%d_w_valid <= 1;"
              % (i, i, i, i),
              "                @(posedge clk); while (!(s%d_aw_ready & s%d_w_ready)) @(posedge clk);" % (i, i),
              "                s%d_aw_valid <= 0; s%d_w_valid <= 0; s%d_b_ready <= 1;" % (i, i, i),
              "                @(posedge clk); while (!s%d_b_valid) @(posedge clk);" % i,
              "                if (s%d_b_data != 0) begin errors = errors + 1; $display(\"ERROR port %d bresp %%0d\", s%d_b_data); end"
              % (i, i, i),
              "                s%d_b_ready <= 0; model[r] = v;" % i,
              "            end else begin",
              "                @(posedge clk); s%d_ar_data <= r << 2; s%d_ar_valid <= 1;" % (i, i),
              "                @(posedge clk); while (!s%d_ar_ready) @(posedge clk);" % i,
              "                s%d_ar_valid <= 0; s%d_r_ready <= 1;" % (i, i),
              "                @(posedge clk); while (!s%d_r_valid) @(posedge clk);" % i,
              "                if (s%d_r_data !== {2'b00, model[r]}) begin errors = errors + 1;" % i,
              "                    $display(\"ERROR port %d reg %%0d got %%h expected %%h\", r, s%d_r_data, model[r]); end" % (i, i),
              "                s%d_r_ready <= 0;" % i,
              "            end",
              "            ops = ops + 1;",
              "        end",
              "        done = done + 1;",
              "    end"]
    T += ["    initial begin",
          "        repeat (10) @(posedge clk); rst = 1;",
          "        wait (done == %d);" % n,
          "        $display(\"RESULT ops %0d errors %0d time %0t\", ops, errors, $time);",
          "        $finish;",
          "    end",
          "    initial begin #50000000; $display(\"RESULT timeout ops %0d errors %0d\", ops, errors); $finish; end",
          "endmodule"]
    return "\n".join(V) + "\n" + "\n".join(T) + "\n"
