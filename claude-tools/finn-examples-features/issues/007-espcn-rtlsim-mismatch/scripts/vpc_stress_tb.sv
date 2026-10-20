// Stress test: dwc_axi (vpc) at the espcn width pairs; bit-sequence must be preserved.
module vpc_tb;
	localparam int unsigned  N = 15;
	localparam int unsigned  IB[N] = '{ 8, 24, 8, 16, 128, 96, 8, 16, 128, 48, 4, 16, 32, 4608, 57 };
	localparam int unsigned  OB[N] = '{ 24, 600, 16, 128, 576, 24, 16, 128, 576, 12, 16, 32, 128, 576, 19 };
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin repeat(8) @(posedge clk); rst <= 0; end
	bit [N-1:0]  done = '0;
	always_comb if(&done) $finish;
	for(genvar t = 0; t < N; t++) begin : genT
		localparam int unsigned  IBITS = IB[t], OBITS = OB[t];
		localparam int unsigned  AI = (IBITS+7)/8*8, AO = (OBITS+7)/8*8;
		// shared bit sequence: bit k of the stream = lfsr-ish hash of k
		function automatic logic bitval(input int unsigned k);
			return  ((k * 2654435761) >> 13) & 1;
		endfunction
		logic [AI-1:0]  idat;
		logic  ivld = 0;
		uwire  irdy;
		uwire [AO-1:0]  odat;
		uwire  ovld;
		logic  ordy = 0;
		dwc_axi #(.IBITS(IBITS), .OBITS(OBITS)) dut (
			.ap_clk(clk), .ap_rst_n(!rst),
			.s_axis_tready(irdy), .s_axis_tvalid(ivld), .s_axis_tdata(idat),
			.m_axis_tready(ordy), .m_axis_tvalid(ovld), .m_axis_tdata(odat)
		);
		int unsigned  cyc = 0;
		int unsigned  pin = 0, pout = 0, nerr = 0;
		always_comb begin
			idat = '0;
			for(int unsigned b = 0; b < IBITS; b++)  idat[b] = bitval(pin*IBITS + b);
		end
		always_ff @(posedge clk) begin
			if(!rst) begin
				automatic int unsigned  ph = (cyc / 3000) % 8;
				cyc <= cyc + 1;
				if(ivld && irdy)  pin <= pin + 1;
				case(ph)
					0, 3, 6: ivld <= 1;
					1:       ivld <= ($urandom % 4 == 0);
					2:       ivld <= ($urandom % 2 == 0);
					4:       ivld <= ($urandom % 8 != 0);
					5:       ivld <= ((cyc % 64) < 3);
					default: ivld <= ($urandom % 3 != 0);
				endcase
				if(ovld && ordy) begin
					for(int unsigned b = 0; b < OBITS; b++) begin
						if(odat[b] !== bitval(pout*OBITS + b)) begin
							nerr <= nerr + 1;
							if(nerr < 5) $display("ERR %0d->%0d: out #%0d bit %0d got %b at cyc %0d", IBITS, OBITS, pout, b, odat[b], cyc);
						end
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
				if(cyc == 200000) begin
					$display("dwc %0d->%0d: in %0d out %0d, %0d errors", IBITS, OBITS, pin, pout, nerr);
					done[t] <= 1;
				end
			end
		end
	end
endmodule
