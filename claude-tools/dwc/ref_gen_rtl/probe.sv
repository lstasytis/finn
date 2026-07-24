`timescale 1ns/1ps
module probe;
  logic clk=0, rst=1; always #1.667 clk=~clk;
  localparam IB=8, OB=4, NI=2, NO=4;
  logic irdy,ivld,ordy,ovld; logic [IB-1:0] idat; logic [OB-1:0] odat;
  dwc_generalized #(.IBITS(IB),.OBITS(OB),.NUM_IN(NI),.NUM_OUT(NO))
    dut(.clk,.rst,.irdy,.ivld,.idat,.ordy,.ovld,.odat);
  int di=0;
  initial begin
    ivld=0; ordy=1; idat=0;
    repeat(4) @(posedge clk); rst<=0; @(posedge clk);
    // feed 2 input words: 0xA5, 0x3C
    for(int f=0; f<2; f++) begin
      // send NI words per frame
      ivld<=1;
      idat<= (f==0)? 8'hA5 : 8'h3C;
      @(posedge clk); while(!irdy) @(posedge clk);
      idat<= (f==0)? 8'h3C : 8'hA5;
      @(posedge clk); while(!irdy) @(posedge clk);
      ivld<=0;
      @(posedge clk);
    end
  end
  int cyc=0;
  always @(posedge clk) if(!rst) begin
    cyc++;
    $display("cyc=%0d ivld=%b irdy=%b idat=%h | ovld=%b ordy=%b odat=%h", cyc,ivld,irdy,idat,ovld,ordy,odat);
    if(cyc>30) $finish;
  end
endmodule
