`timescale 1ns/1ps
module tput;
  logic clk=0,rst=1; always #1.667 clk=~clk;
  // downscale 96->40 (NI=5,NO=12, ideal 17/frame) and upscale 40->96 (NI=12? use 5->2 style)
  parameter IB=96,OB=40,NI=5,NO=12;
  logic irdy,ivld,ordy,ovld; logic [IB-1:0] idat; logic [OB-1:0] odat;
  dwc_generalized #(IB,OB,NI,NO) d(.clk,.rst,.irdy,.ivld,.idat,.ordy,.ovld,.odat);
  int icnt=0,ocnt=0,cyc=0; assign ivld=1; assign idat=cyc; assign ordy=1;
  initial begin repeat(4)@(posedge clk); rst<=0;
    repeat(2000) @(posedge clk);
    $display("cycles=2000 in_beats=%0d out_beats=%0d  frames_out=%0d expect_cyc/frame~%0d actual_cyc/frame=%0.2f",
      icnt,ocnt, ocnt/NO, NI+NO, 2000.0/(ocnt/NO));
    $finish; end
  always @(posedge clk) if(!rst) begin cyc++; if(ivld&&irdy)icnt++; if(ovld&&ordy)ocnt++; end
endmodule
