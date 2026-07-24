// Self-checking testbench for vpc.sv: verifies that, with no length change,
// the module re-packs the SAME N-element vector from PI lanes/beat to PO
// lanes/beat (little-endian). Golden model: the flat element sequence is the
// identity. Exercises input- and output-side backpressure. A clocking block
// samples DUT outputs in the preponed region so the AXIS handshake is race-free.
// Parameters overridable via -generic_top.
`timescale 1ns/1ps
module vpc_tb #(
	int unsigned  W  = 8,
	int unsigned  N  = 16,
	int unsigned  PI = 4,
	int unsigned  PO = 2,
	int unsigned  VECS = 3,     // how many vectors to stream back-to-back
	bit           STALL = 1     // enable pseudo-random backpressure
)();
	localparam int unsigned  TRNI = (N + PI - 1)/PI;   // input beats / vector
	localparam int unsigned  TRNO = (N + PO - 1)/PO;   // output beats / vector

	logic  clk = 0, rst = 1;
	always #5 clk = ~clk;

	logic [PI-1:0][W-1:0]  idat;
	logic  ivld, irdy;
	logic [PO-1:0][W-1:0]  odat;
	logic  ovld, ordy;

	vpc #(.W(W), .N(N), .PI(PI), .PO(PO)) dut (
		.clk(clk), .rst(rst),
		.idat(idat), .ivld(ivld), .irdy(irdy),
		.odat(odat), .ovld(ovld), .ordy(ordy)
	);

	// Sample DUT outputs (irdy/ovld/odat) in the preponed region; drive stimulus
	// with #0 output skew.
	clocking cb @(posedge clk);
		default input #1step output #0;
		input  irdy, ovld, odat;
		output ivld, idat, ordy;
	endclocking

	// Golden element value for vector v, element index e (non-zero so stray
	// zeros are detectable), truncated to W bits.
	function automatic logic [W-1:0] gold(int v, int e);
		return (v*N + e + 1);
	endfunction

	int unsigned  errors = 0;
	int unsigned  lfsr = 32'h1;
	function automatic bit rnd();  // cheap pseudo-random backpressure
		lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & 32'hD0000001);
		return STALL && lfsr[0];
	endfunction

	// ---- Input feeder ----
	initial begin
		cb.ivld <= 0; cb.idat <= '0;
		wait(!rst); @(cb);
		for(int v = 0; v < VECS; v++) begin
			for(int b = 0; b < TRNI; b++) begin
				automatic logic [PI-1:0][W-1:0]  beat;
				for(int p = 0; p < PI; p++) begin
					automatic int e = b*PI + p;
					beat[p] = (e < N) ? gold(v, e) : '0;  // zero-pad excess lanes
				end
				// optional stall with ivld low
				while(rnd()) begin cb.ivld <= 0; @(cb); end
				// present beat, hold until accepted (ivld && irdy sampled)
				cb.ivld <= 1; cb.idat <= beat;
				@(cb);
				while(!cb.irdy) @(cb);
				cb.ivld <= 0;
			end
		end
		cb.ivld <= 0;
	end

	// ---- Output collector + checker ----
	initial begin
		cb.ordy <= 0;
		wait(!rst); @(cb);
		for(int v = 0; v < VECS; v++) begin
			for(int b = 0; b < TRNO; b++) begin
				// optional stall with ordy low
				while(rnd()) begin cb.ordy <= 0; @(cb); end
				// assert ready, wait for a valid beat (transfer at that edge)
				cb.ordy <= 1; @(cb);
				while(!cb.ovld) @(cb);
				for(int p = 0; p < PO; p++) begin
					automatic int e = b*PO + p;
					automatic logic [W-1:0] exp = (e < N) ? gold(v, e) : '0;
					if(cb.odat[p] !== exp) begin
						$display("MISMATCH v=%0d beat=%0d lane=%0d idx=%0d got=%0h exp=%0h",
							v, b, p, e, cb.odat[p], exp);
						errors++;
					end
				end
				cb.ordy <= 0;
			end
		end
		cb.ordy <= 0;
		#20;
		if(errors == 0) $display("VPC_TB_PASS  W=%0d N=%0d PI=%0d PO=%0d VECS=%0d", W, N, PI, PO, VECS);
		else            $display("VPC_TB_FAIL  W=%0d N=%0d PI=%0d PO=%0d errors=%0d", W, N, PI, PO, errors);
		$finish;
	end

	initial begin
		repeat(4) @(posedge clk);
		rst = 0;
		#200000;  // global timeout guard
		$display("VPC_TB_FAIL  TIMEOUT W=%0d N=%0d PI=%0d PO=%0d", W, N, PI, PO);
		$finish;
	end
endmodule
