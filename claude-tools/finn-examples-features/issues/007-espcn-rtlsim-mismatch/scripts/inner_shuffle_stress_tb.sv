// Stress test: inner_shuffle (I x J) -> (J x I) per page with random valid/ready phases.
module ish_tb;
	localparam int unsigned  BITS = 32, I = 256, J = 3, SIMD = 1;
	localparam int unsigned  PAGE = I*J;
	localparam int unsigned  NPAGES = 40;
	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin repeat(8) @(posedge clk); rst <= 0; end

	logic [BITS-1:0]  idat = 0;
	logic  ivld = 0;
	uwire  irdy;
	uwire [BITS-1:0]  odat;
	uwire  ovld;
	logic  ordy = 0;
	inner_shuffle #(.BITS(BITS), .I(I), .J(J), .SIMD(SIMD)) dut (
		.clk, .rst, .irdy, .ivld, .idat, .ordy, .ovld, .odat
	);
	int unsigned  cyc = 0;
	int unsigned  pin = 0, pout = 0, nerr = 0;
	function automatic logic [BITS-1:0] expected(input int unsigned k);
		// output order: page p, j, i  <-  input index p*PAGE + i*J + j
		automatic int unsigned  p = k / PAGE, r = k % PAGE, j = r / I, i = r % I;
		return  BITS'(p*PAGE + i*J + j);
	endfunction
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
				5:       ivld <= ((cyc % 64) < 3);      // sparse bursts
				default: ivld <= ($urandom % 3 != 0);
			endcase
			if(ovld && ordy) begin
				if(odat !== expected(pout)) begin
					nerr <= nerr + 1;
					if(nerr < 20) $display("ERR: out #%0d got %h exp %h at cyc %0d", pout, odat, expected(pout), cyc);
				end
				pout <= pout + 1;
			end
			case(ph)
				0:       ordy <= 0;
				1:       ordy <= 1;
				2:       ordy <= ($urandom % 3 == 0);
				3:       ordy <= ($urandom % 2 == 0);
				4:       ordy <= ($urandom % 9 != 0);
				5:       ordy <= ((cyc % 100) < 40);    // long stalls
				6:       ordy <= ((cyc % 7) == 0);
				default: ordy <= ($urandom % 5 == 0);
			endcase
			if(pout == NPAGES*PAGE) begin
				$display("inner_shuffle I=%0d J=%0d: %0d outputs, %0d errors, %0d cycles", I, J, pout, nerr, cyc);
				$finish;
			end
			if(cyc > 2000000) begin $display("TIMEOUT pin=%0d pout=%0d", pin, pout); $finish; end
		end
	end
endmodule
