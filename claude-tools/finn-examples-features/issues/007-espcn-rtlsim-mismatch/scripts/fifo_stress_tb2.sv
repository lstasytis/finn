// Stress test: fifo.sv at depths the previous session did not cover; random valid/ready phases.
module fifo_tb;
	localparam int unsigned  W = 32;
	localparam int unsigned  N = 10;
	localparam int unsigned  DEPTHS[N] = '{ 2, 3, 5, 6, 35, 39, 230, 256, 272, 513 };
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin repeat(8) @(posedge clk); rst <= 0; end
	bit [N-1:0]  done = '0;
	always_comb if(&done) $finish;
	for(genvar t = 0; t < N; t++) begin : genT
		logic [W-1:0]  idat = 0;
		logic  ivld = 0;
		uwire  irdy;
		uwire [W-1:0]  odat;
		uwire  ovld;
		logic  ordy = 0;
		fifo #(.DEPTH(DEPTHS[t]), .DATA_WIDTH(W)) dut (
			.clk, .rst, .idat, .ivld, .irdy, .odat, .ovld, .ordy, .count(), .maxcount()
		);
		int unsigned  cyc = 0;
		int unsigned  pin = 0, pout = 0, nerr = 0;
		always_ff @(posedge clk) begin
			if(!rst) begin
				automatic int unsigned  ph = (cyc / 3000) % 8;
				cyc <= cyc + 1;
				if(ivld && irdy) begin pin <= pin + 1; idat <= idat + 1; end
				case(ph)
					0, 3, 6: ivld <= 1;
					1:       ivld <= ($urandom % 4 == 0);
					2:       ivld <= ($urandom % 2 == 0);
					4:       ivld <= ($urandom % 8 != 0);
					5:       ivld <= ((cyc % 64) < 3);
					default: ivld <= ($urandom % 3 != 0);
				endcase
				if(ovld && ordy) begin
					if(odat !== W'(pout)) begin
						nerr <= nerr + 1;
						if(nerr < 10) $display("ERR depth %0d: got %h exp %h at cyc %0d", DEPTHS[t], odat, W'(pout), cyc);
					end
					pout <= pout + 1;
				end
				case(ph)
					0:       ordy <= 0;
					1:       ordy <= 1;
					2:       ordy <= ($urandom % 3 == 0);
					3:       ordy <= ($urandom % 2 == 0);
					4:       ordy <= ($urandom % 9 != 0);
					5:       ordy <= ((cyc % 100) < 40);
					6:       ordy <= ((cyc % 7) == 0);
					default: ordy <= ($urandom % 5 == 0);
				endcase
				if(cyc == 300000) begin
					$display("fifo depth %0d: %0d pops, %0d errors", DEPTHS[t], pout, nerr);
					done[t] <= 1;
				end
			end
		end
	end
endmodule
