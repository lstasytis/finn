# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stream links between overlay tiles (DFX partitions) through static anchor registers.

A stream (data, valid, ready) from a producer tile to a consumer tile crosses the static design
as plain wires with one anchor register each (RapidStream's anchors, DW's relay stations), so that
every timing path ends at a static register and every tile is timed completely on its own:

    producer tile                static                   consumer tile
    rwi_link_tx  --data/valid--> [FF] --data/valid-->    rwi_link_rx (FIFO)
                 <----ready----- [FF] <----ready-----

The ready seen by the producer is late (registered in the receiver, the anchor and the
transmitter), so the receiver signals ready only while its FIFO can absorb the words already in
flight: depth 8, ready while at most DEPTH - SLACK entries are used. Throughput: one word per cycle
while the consumer takes one per cycle (the FIFO never runs into the ready threshold).
"""

RX_DEPTH = 8
# words that can arrive after the receiver deasserts ready: its own ready register, the anchor
# on ready, the transmitter's register on ready and the anchor + transmitter registers on data
RX_SLACK = 5

LINK_TX = """module rwi_link_tx #(parameter W = 1) (
    input clk,
    input rst,
    // from the tile's logic
    input [W-1:0] s_data,
    input s_valid,
    output s_ready,
    // to the link (static anchors)
    output reg [W-1:0] l_data,
    output reg l_valid,
    input l_ready
);
    // l_ready: the receiver's permission (late); a word is sent in every cycle in which it holds
    reg rdy = 1'b0;
    assign s_ready = rdy;
    always @(posedge clk) begin
        rdy <= rst & l_ready;
        l_valid <= rst & s_valid & rdy;
        l_data <= s_data;
    end
endmodule
"""

LINK_RX = """module rwi_link_rx #(parameter W = 1, parameter DEPTH = %d, parameter SLACK = %d) (
    input clk,
    input rst,
    // from the link (static anchors)
    input [W-1:0] l_data,
    input l_valid,
    output reg l_ready,
    // to the tile's logic
    output [W-1:0] m_data,
    output m_valid,
    input m_ready
);
    reg [W-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH):0] wp = 0, rp = 0;
    wire [$clog2(DEPTH):0] used = wp - rp;
    assign m_valid = used != 0;
    assign m_data = mem[rp[$clog2(DEPTH)-1:0]];
    always @(posedge clk) begin
        if (!rst) begin
            wp <= 0;
            rp <= 0;
            l_ready <= 1'b0;
        end else begin
            if (l_valid) begin
                mem[wp[$clog2(DEPTH)-1:0]] <= l_data;
                wp <= wp + 1;
            end
            if (m_valid && m_ready) rp <= rp + 1;
            l_ready <= used + (l_valid ? 1 : 0) <= DEPTH - SLACK;
        end
    end
endmodule
""" % (RX_DEPTH, RX_SLACK)

ANCHOR = """module rwi_anchor #(parameter W = 1) (
    input clk,
    input [W-1:0] d,
    output reg [W-1:0] q
);
    always @(posedge clk) q <= d;
endmodule
"""


def link_testbench(width=16, n=2000, seed=1):
    """xsim testbench: tx -> anchor -> rx with random producer stalls and consumer backpressure;
    checks order, no loss / duplication, and reports the cycles per word at full rate."""
    return """`timescale 1ns/1ps
module tb;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 0;
    localparam W = %(w)d, N = %(n)d;
    reg [W-1:0] s_data = 0; reg s_valid = 0; wire s_ready;
    wire [W-1:0] l_data, a_data; wire l_valid, a_valid, l_ready, a_ready;
    wire [W-1:0] m_data; wire m_valid; reg m_ready = 0;
    rwi_link_tx #(.W(W)) tx (.clk(clk), .rst(rst), .s_data(s_data), .s_valid(s_valid), .s_ready(s_ready),
        .l_data(l_data), .l_valid(l_valid), .l_ready(a_ready));
    rwi_anchor #(.W(W + 1)) af (.clk(clk), .d({l_valid, l_data}), .q({a_valid, a_data}));
    rwi_anchor #(.W(1)) ab (.clk(clk), .d(l_ready), .q(a_ready));
    rwi_link_rx #(.W(W)) rx (.clk(clk), .rst(rst), .l_data(a_data), .l_valid(a_valid), .l_ready(l_ready),
        .m_data(m_data), .m_valid(m_valid), .m_ready(m_ready));
    integer sent = 0, got = 0, errors = 0, cyc = 0, phase = 0, t_full0 = 0, t_full1 = 0;
    integer seed = %(seed)d;
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (rst) begin
            // producer: a new word whenever the previous one was taken
            if (s_valid && s_ready) begin sent = sent + 1; s_data <= sent; end
            if (phase == 0) s_valid <= ($random(seed) %% 4) != 0;      // random stalls
            else s_valid <= 1;                                          // full rate
            if (phase == 0) m_ready <= ($random(seed) %% 3) != 0;      // random backpressure
            else m_ready <= 1;
            if (m_valid && m_ready) begin
                if (m_data !== got[W-1:0]) begin errors = errors + 1; if (errors < 5) $display("ERROR got %%0d expected %%0d", m_data, got); end
                got = got + 1;
                if (got == N / 2) begin phase = 1; t_full0 = cyc; end
                if (got == N) begin t_full1 = cyc; end
            end
        end
    end
    initial begin
        repeat (5) @(posedge clk); rst = 1;
        wait (got == N);
        $display("RESULT words %%0d errors %%0d full_rate_cycles_per_word %%0.3f", got, errors, (t_full1 - t_full0) * 1.0 / (N / 2));
        $finish;
    end
endmodule
""" % {"w": width, "n": n, "seed": seed}
