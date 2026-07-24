// Self-checking TB for dwc_generalized. Precomputes random input beats and the
// golden output beats (flat little-endian bit sequence, zero-pad / crop), then
// drives/checks with independent random backpressure on both AXIS interfaces.
`timescale 1ns/1ps

module dwc_check #(
  int unsigned IBITS, int unsigned OBITS,
  int unsigned NUM_IN, int unsigned NUM_OUT,
  int unsigned NFRAMES = 4, int unsigned SEED = 1
)(input logic clk, input logic rst, output logic done, output logic fail);

  localparam int unsigned NI = NFRAMES*NUM_IN;
  localparam int unsigned NO = NFRAMES*NUM_OUT;

  logic irdy, ivld, ordy, ovld;
  logic [IBITS-1:0] idat;
  logic [OBITS-1:0] odat;

  dwc_generalized #(.IBITS(IBITS), .OBITS(OBITS), .NUM_IN(NUM_IN), .NUM_OUT(NUM_OUT))
    dut(.clk(clk), .rst(rst), .irdy(irdy), .ivld(ivld), .idat(idat),
        .ordy(ordy), .ovld(ovld), .odat(odat));

  logic [IBITS-1:0] in_words  [NI];
  logic [OBITS-1:0] exp_words [NO];

  // reference generation
  initial begin
    int unsigned s = SEED;
    for(int unsigned i = 0; i < NI; i++) begin
      logic [IBITS-1:0] w = '0;
      for(int unsigned b = 0; b < IBITS; b++) begin
        s = s*1103515245 + 12345; w[b] = s[31];
      end
      in_words[i] = w;
    end
    // per frame, build flat input bits and slice output beats (pad 0 / crop)
    for(int unsigned f = 0; f < NFRAMES; f++) begin
      for(int unsigned w = 0; w < NUM_OUT; w++) begin
        logic [OBITS-1:0] ow = '0;
        for(int unsigned b = 0; b < OBITS; b++) begin
          int unsigned j = w*OBITS + b;               // global output bit index
          if(j < NUM_IN*IBITS)
            ow[b] = in_words[f*NUM_IN + (j/IBITS)][j%IBITS];
        end
        exp_words[f*NUM_OUT + w] = ow;
      end
    end
  end

  // driver: standard AXIS master, always-valid (input backpressure comes from
  // the DUT's irdy). di advances only on an accepted beat.
  int unsigned di = 0;
  always_ff @(posedge clk) begin
    if(rst) di <= 0;
    else if(ivld && irdy) di <= di + 1;
  end
  assign ivld = (di < NI);
  assign idat = in_words[(di < NI)? di : 0];

  // checker: random output backpressure (ordy toggles freely - always legal).
  int unsigned ci = 0;
  int unsigned crng = SEED ^ 32'h1234;
  always_ff @(posedge clk) begin
    if(rst) begin ci <= 0; ordy <= 0; fail <= 0; done <= 0; end
    else begin
      crng = crng*1103515245 + 12345;
      ordy <= crng[27];                  // random backpressure
      if(ovld && ordy) begin
        if(odat !== exp_words[ci]) begin
          fail <= 1;
          $display("  MISMATCH cfg %0d->%0d NI=%0d NO=%0d idx=%0d got=%h exp=%h",
                   IBITS, OBITS, NUM_IN, NUM_OUT, ci, odat, exp_words[ci]);
        end
        if(ci == NO-1) done <= 1;
        ci <= ci + 1;
      end
    end
  end
endmodule

module dwc_gen_tb;
  logic clk = 0, rst = 1;
  always #1.667 clk = ~clk;   // ~300MHz

  localparam int unsigned NC = 16;
  logic [NC-1:0] done, fail;

  // configs: {IBITS,OBITS,NUM_IN,NUM_OUT}
  dwc_check #(8,4,2,4)      c0 (clk,rst,done[0],fail[0]);   // mult down
  dwc_check #(4,8,4,2)      c1 (clk,rst,done[1],fail[1]);   // mult up
  dwc_check #(6,4,2,3)      c2 (clk,rst,done[2],fail[2]);   // non-mult down
  dwc_check #(4,6,3,2)      c3 (clk,rst,done[3],fail[3]);   // non-mult up
  dwc_check #(7,5,5,7)      c4 (clk,rst,done[4],fail[4]);   // coprime
  dwc_check #(5,7,7,5)      c5 (clk,rst,done[5],fail[5]);   // coprime
  dwc_check #(4,13,6,2)     c6 (clk,rst,done[6],fail[6]);   // pad (out>in)
  dwc_check #(10,4,2,2)     c7 (clk,rst,done[7],fail[7]);   // crop (out<in)
  dwc_check #(96,40,2,5)    c8 (clk,rst,done[8],fail[8]);   // wide non-mult down
  dwc_check #(40,96,5,2)    c9 (clk,rst,done[9],fail[9]);   // wide non-mult up
  dwc_check #(44,48,1,1)    c10(clk,rst,done[10],fail[10]); // pad non-mult
  dwc_check #(128,64,2,4)   c11(clk,rst,done[11],fail[11]); // wide mult down
  dwc_check #(48,32,4,7)    c12(clk,rst,done[12],fail[12]); // crop non-mult
  dwc_check #(480,512,1,1)  c13(clk,rst,done[13],fail[13]); // wide near-1:1 pad
  dwc_check #(1024,768,3,4) c14(clk,rst,done[14],fail[14]); // wide down non-mult
  dwc_check #(8,64,8,1)     c15(clk,rst,done[15],fail[15]); // high-ratio up

  initial begin
    repeat(4) @(posedge clk);
    rst <= 0;
    fork begin : timeout repeat(200000) @(posedge clk); $display("TIMEOUT done=%b", done); end join_none
    wait(&done);
    disable timeout;
    repeat(5) @(posedge clk);
    if(|fail) $display("\nRESULT: FAIL (fail mask=%b)", fail);
    else      $display("\nRESULT: ALL PASS (%0d configs)", NC);
    $finish;
  end
endmodule
