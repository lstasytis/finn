`timescale 1ns/1ps
module probe;
  logic clk=0, rst=1; always #1.667 clk=~clk;
  localparam IB=8, OB=4, NI=2, NO=4;
  logic irdy,ivld=0,ordy=0; logic [IB-1:0] idat=0; logic [OB-1:0] odat; logic ovld;
  dwc_generalized #(.IBITS(IB),.OBITS(OB),.NUM_IN(NI),.NUM_OUT(NO))
    dut(.clk,.rst,.irdy,.ivld,.idat,.ordy,.ovld,.odat);
  initial begin
    repeat(4)@(posedge clk); rst<=0;
    @(posedge clk); ivld<=1; idat<=8'hA5; ordy<=1;
    @(posedge clk); idat<=8'h3C;
    @(posedge clk); ivld<=0;
    repeat(10)@(posedge clk); $finish;
  end
  int c=0;
  always @(posedge clk) if(!rst) begin c++;
    $display("c=%0d itrn(iv&ir)=%b%b idat=%h | otrn(ov&or)=%b%b odat=%h || ITrn=%0d OTrn=%0d ICap=%0d ORdy=%0d Buf0=%h Buf1=%h Buf2=%h",
      c, ivld,irdy, idat, ovld,ordy, odat, dut.ITrn, dut.OTrn, dut.ICap, dut.ORdy, dut.Buf[0], dut.Buf[1], dut.Buf[2]);
  end
endmodule
