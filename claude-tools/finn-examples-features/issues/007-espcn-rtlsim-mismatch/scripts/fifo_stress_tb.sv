// Stress test: high-rate push, random pop with phases; check data order and X.
module tb2;
	localparam int unsigned  W = 16;
	localparam int unsigned  N = 6;
	localparam int unsigned  DEPTHS[N] = '{ 2094, 2912, 2190, 5587, 1060, 510 };
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
		int unsigned  pin = 0, pout = 0;
		always_ff @(posedge clk) begin
			if(!rst) begin
				automatic int unsigned  ph = (cyc / 5000) % 6;
				cyc <= cyc + 1;
				// push side
				if(ivld && irdy) begin pin <= pin + 1; idat <= idat + 1; end
				case(ph)
					0, 3:    ivld <= 1;
					1:       ivld <= ($urandom % 4 == 0);
					2:       ivld <= ($urandom % 2 == 0);
					default: ivld <= ($urandom % 8 != 0);
				endcase
				// pop side
				if(ovld && ordy) begin
					assert(odat === W'(pout)) else begin
						$error("depth %0d: got %h exp %h at cyc %0d", DEPTHS[t], odat, W'(pout), cyc);
						$stop;
					end
					pout <= pout + 1;
				end
				case(ph)
					0:       ordy <= 0;           // fill
					1:       ordy <= 1;
					2:       ordy <= ($urandom % 3 == 0);
					3:       ordy <= ($urandom % 2 == 0);
					4:       ordy <= ($urandom % 9 != 0);
					default: ordy <= ($urandom % 5 == 0);
				endcase
				if(cyc == 400000) begin
					$display("depth %0d ok: %0d pops", DEPTHS[t], pout);
					done[t] <= 1;
				end
			end
		end
	end
endmodule
