// Issue 007 reproducer: mvu_vvu_axi output queue overflow under output backpressure.
// Default parameters = espcn fold-target MVAU_rtl_0 (MW=SIMD=75 -> SF=1, PE=2, MH=64, 8x8 bit,
// DSP48E2). Inputs and weights always valid, output tready random (+READY_PCT, default 50).
// Checks every output against a dot-product reference and counts X words / mismatches.
// Plusargs: +READY_PCT=<0..100>, +CYCLES=<n>, +BURST (stall bursts of 1..12 cycles instead),
// +REPLAY=<file>: sink behind a 2-deep FIFO pops only at the recorded cycles (one per line,
// relative to the first output word; from the full-design tap of the FIFO after the MVU).
// Defines: TB_MW, TB_SIMD, TB_VER (DSP version), TB_STRUCT (structural DSP primitives).
// Run: mvu_queue_run.sh <workdir> <tag> orig|<patched axi.sv> [defines] -- [-testplusarg ...]
module mvu_queue_tb;
	localparam int unsigned  MW = `ifdef TB_MW `TB_MW `else 75 `endif;
	localparam int unsigned  SIMD = `ifdef TB_SIMD `TB_SIMD `else 75 `endif;
	localparam int unsigned  MH = 64;
	localparam int unsigned  PE = 2;
	localparam int unsigned  WW = `ifdef TB_WW `TB_WW `else 8 `endif, AW = `ifdef TB_AW `TB_AW `else 8 `endif, ACCU = 20;
	localparam int unsigned  SF = MW/SIMD;
	localparam int  WMAX = (1<<(WW-1))-1 < 15 ? (1<<(WW-1))-1 : 15;

	logic  clk = 0;
	always #5ns clk = !clk;
	logic  rst = 1;
	initial begin repeat(16) @(posedge clk); rst <= 0; end

	logic [PE*SIMD*WW-1:0]  wdat;
	uwire  wrdy;
	logic [SIMD*AW-1:0]  adat;
	logic  avld = 0;
	uwire  ardy;
	uwire [(PE*ACCU+7)/8*8-1:0]  odat;
	uwire  ovld;
	logic  ordy = 0;

	mvu_vvu_axi #(
		.IS_MVU(1), .VERSION(`ifdef TB_VER `TB_VER `else 2 `endif), .MW(MW), .MH(MH), .PE(PE), .SIMD(SIMD),
		.ACTIVATION_WIDTH(AW), .WEIGHT_WIDTH(WW), .ACCU_WIDTH(ACCU), .NARROW_WEIGHTS(1),
		.SIGNED_ACTIVATIONS(0), .SEGMENTLEN(8), .FORCE_BEHAVIORAL(`ifdef TB_STRUCT 0 `else 1 `endif)
	) dut (
		.ap_clk(clk), .ap_clk2x(clk), .ap_rst_n(!rst),
		.s_axis_weights_tdata(wdat), .s_axis_weights_tvalid(!rst), .s_axis_weights_tready(wrdy),
		.s_axis_input_tdata(adat), .s_axis_input_tvalid(avld), .s_axis_input_tready(ardy),
		.m_axis_output_tdata(odat), .m_axis_output_tvalid(ovld), .m_axis_output_tready(ordy)
	);

	// small random values keep the 20-bit accumulator in range
	function automatic logic [SIMD*AW-1:0] rnd_a();
		for(int s = 0; s < SIMD; s++)  rnd_a[s*AW +: AW] = $urandom_range(0, (1<<AW)-1 < 15 ? (1<<AW)-1 : 15);
	endfunction
	function automatic logic [PE*SIMD*WW-1:0] rnd_w();
		for(int s = 0; s < PE*SIMD; s++)  rnd_w[s*WW +: WW] = $signed($urandom_range(0, 2*WMAX)) - WMAX;
	endfunction

	int unsigned  ready_pct = 50, cycles = 200000;
	bit  burst = 0;
	string  replay_file;
	bit  replay = 0;
	bit  pop_at[int];
	int  fcnt = 0, base = -1;
	bit  o_hs = 0;
	int  exp_q[$];
	int  partial[PE];
	int unsigned  sf_cnt = 0;
	int unsigned  n_out = 0, n_x = 0, n_bad = 0, first_x = 0, n_stall = 0;
	int  optr_max = -1;

	initial begin
		void'($value$plusargs("READY_PCT=%d", ready_pct));
		void'($value$plusargs("CYCLES=%d", cycles));
		burst = $test$plusargs("BURST");
		if($value$plusargs("REPLAY=%s", replay_file)) begin
			automatic int  fd = $fopen(replay_file, "r"), c;
			replay = 1;
			while($fscanf(fd, "%d", c) == 1)  pop_at[c] = 1;
			$fclose(fd);
		end
		wdat = rnd_w();
		adat = rnd_a();
		for(int p = 0; p < PE; p++)  partial[p] = 0;
	end

	// sample handshakes at negedge (stable), drive at posedge
	int unsigned  cyc = 0, stall_left = 0;
	bit  w_hs = 0, a_hs = 0;
	always @(negedge clk) begin
		w_hs = !rst && wrdy;
		a_hs = !rst && avld && ardy;
		if(w_hs) begin
			// weight handshake = one SIMD chunk of one neuron fold: build the reference
			for(int p = 0; p < PE; p++) begin
				automatic int  acc = 0;
				for(int s = 0; s < SIMD; s++)
					acc += $signed(wdat[(p*SIMD+s)*WW +: WW]) * int'(dut.amvau[s]);
				partial[p] += acc;
			end
			if(++sf_cnt == SF) begin
				sf_cnt = 0;
				for(int p = 0; p < PE; p++) begin
					exp_q.push_back(partial[p]);
					partial[p] = 0;
				end
			end
		end
		o_hs = !rst && ovld && ordy;
		if(o_hs) begin
			n_out++;
			for(int p = 0; p < PE; p++) begin
				automatic logic [ACCU-1:0]  g = odat[p*ACCU +: ACCU];
				automatic int  e = exp_q.pop_front();
				if($isunknown(g)) begin
					if(p == 0) begin
						if(n_x == 0)  first_x = n_out;
						n_x++;
					end
				end
				else if($signed(g) != e) begin
					if(p == 0)  n_bad++;
				end
			end
		end
		if(!rst && int'($signed(dut.blkOutput.OPtr)) > optr_max)  optr_max = $signed(dut.blkOutput.OPtr);
	end
	always @(posedge clk) begin
		cyc <= cyc + 1;
		if(!rst) begin
			avld <= 1;
			if(a_hs)  adat <= rnd_a();
			if(w_hs)  wdat <= rnd_w();
			if(replay) begin
				// 2-deep FIFO between DUT and the recorded sink; time 0 = first output word
				automatic int  n = fcnt;
				if(o_hs) begin
					if(base < 0)  base = cyc;
					n++;
				end
				if(base >= 0 && fcnt > 0 && pop_at.exists(cyc - base))  n--;
				fcnt = n;
				ordy <= n < 2;
			end
			else if(burst) begin
				if(stall_left > 0) begin ordy <= 0; stall_left--; end
				else if($urandom_range(0, 99) < 100 - ready_pct) begin
					ordy <= 0; stall_left = $urandom_range(0, 11); n_stall++;
				end
				else  ordy <= 1;
			end
			else  ordy <= $urandom_range(0, 99) < ready_pct;
		end
		if(cyc == cycles) begin
			$display("RESULT MW=%0d SIMD=%0d SF=%0d ready_pct=%0d burst=%0d outputs=%0d X_words=%0d (first at output %0d) wrong_words=%0d OPtr_max=%0d queue_entries=%0d",
				MW, SIMD, SF, ready_pct, burst, n_out, n_x, first_x, n_bad, optr_max,
				$size(dut.blkOutput.OBuf));
			$finish;
		end
	end
endmodule
