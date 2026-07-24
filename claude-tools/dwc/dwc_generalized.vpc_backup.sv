/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Generalized Stream Data Width Converter (non-multiple + padding).
 *
 * @description
 *  Converts a stream of NUM_IN input beats (IBITS each) per frame into NUM_OUT
 *  output beats (OBITS each), preserving the flat little-endian bit sequence.
 *  Supports ARBITRARY (non-multiple, coprime) width ratios and, when
 *  NUM_IN*IBITS != NUM_OUT*OBITS, zero-PADDING (output longer) or CROPPING
 *  (output shorter). Frames stream back-to-back (no explicit rep count).
 *
 *  Internally normalized by GCD(IBITS,OBITS) to a lane granularity, following
 *  T. Preußer's vpc.sv two-phase (pop-then-push) lane buffer. The push is a
 *  variable-offset placement (crossbar) -- fast (single mux level, 300MHz) but
 *  O(PI0*CAP) in area; a pipelined-barrel variant is planned for wide downscale.
 *
 *  II=1 on both interfaces; no combinational path from ordy to irdy.
 *****************************************************************************/
module dwc_generalized #(
	int unsigned  IBITS,
	int unsigned  OBITS,
	int unsigned  NUM_IN,   // input beats per frame
	int unsigned  NUM_OUT   // output beats per frame
)(
	//- Global Control ------------------
	input	logic  clk,
	input	logic  rst,

	//- AXI Stream - Input --------------
	output	logic  irdy,
	input	logic  ivld,
	input	logic [IBITS-1:0]  idat,

	//- AXI Stream - Output -------------
	input	logic  ordy,
	output	logic  ovld,
	output	logic [OBITS-1:0]  odat
);

	// GCD normalization -> lane granularity
	function automatic int unsigned gcd(input int unsigned a, input int unsigned b);
		return (b == 0)? a : gcd(b, a % b);
	endfunction
	localparam int unsigned  GCD = gcd(IBITS, OBITS);
	localparam int unsigned  W0  = GCD;            // lane width (bits)
	localparam int unsigned  PI0 = IBITS / GCD;    // input lanes / beat
	localparam int unsigned  PO0 = OBITS / GCD;    // output lanes / beat
	localparam int unsigned  CAP = PI0 + PO0;      // buffer capacity (lanes)

	// Frame progress counters (count down: N-1 .. 0 .. -1 = done)
	logic signed [$clog2(NUM_IN)+1:0]   ITrn = NUM_IN-1;
	logic signed [$clog2(NUM_OUT)+1:0]  OTrn = NUM_OUT-1;
	uwire  idone = ITrn[$left(ITrn)];
	uwire  odone = OTrn[$left(OTrn)];
	uwire  itrn  = ivld && irdy;
	uwire  otrn  = ovld && ordy;

	// Lane buffer + occupancy indicators (ORdy>=0: a full output beat is ready;
	// ICap>=0: room for another input beat). Valid lane count = ORdy + PO0.
	logic [W0-1:0]  Buf [CAP];
	logic signed [$clog2(CAP)+1:0]  ICap =  PO0;
	logic signed [$clog2(CAP)+1:0]  ORdy = -PO0;
	uwire  full_beat = !ORdy[$left(ORdy)];

	always_ff @(posedge clk) begin
		if(rst) begin
			Buf  <= '{ default: '0 };
			ICap <=  PO0;
			ORdy <= -PO0;
			ITrn <=  NUM_IN-1;
			OTrn <=  NUM_OUT-1;
		end
		else begin
			automatic logic [W0-1:0]  n_buf [CAP];
			automatic logic signed [$clog2(CAP)+1:0]   n_icap = ICap;
			automatic logic signed [$clog2(CAP)+1:0]   n_ordy = ORdy;
			automatic logic signed [$clog2(NUM_IN)+1:0]   n_itrn = ITrn;
			automatic logic signed [$clog2(NUM_OUT)+1:0]  n_otrn = OTrn;
			n_buf = Buf;

			// Phase 1: pop PO0 lanes from the head, shift down, zero-fill the
			// vacated top lanes (so padding after input-drain emits zeros).
			if(otrn) begin
				n_buf[0 +: CAP-PO0] = n_buf[PO0 +: CAP-PO0];
				for(int unsigned i = CAP-PO0; i < CAP; i++)  n_buf[i] = '0;
				n_icap += PO0;
				n_ordy -= PO0;
				n_otrn--;
			end

			// Phase 2: push PI0 input lanes at the current tail. Once the output
			// frame is complete (cropping) input is consumed but not stored.
			if(itrn) begin
				if(!odone) begin
					// valid lane count = occupancy after any pop this cycle.
					// Keep the add SIGNED (n_ordy can be negative) or it wraps.
					automatic int  ofs = $signed(n_ordy) + $signed(PO0);
					for(int unsigned p = 0; p < PI0; p++)
						n_buf[ofs + p] = idat[p*GCD +: GCD];
					n_icap -= PI0;
					n_ordy += PI0;
				end
				n_itrn--;
			end

			// Start the next frame once both sides are complete.
			if(n_itrn[$left(n_itrn)] && n_otrn[$left(n_otrn)]) begin
				n_itrn = NUM_IN-1;
				n_otrn = NUM_OUT-1;
				n_icap =  PO0;
				n_ordy = -PO0;
			end

			Buf  <= n_buf;
			ICap <= n_icap;
			ORdy <= n_ordy;
			ITrn <= n_itrn;
			OTrn <= n_otrn;
		end
	end

	// Accept input while the input frame is unfinished and there is room -- or
	// the output frame is already done, in which case we drain (crop) the rest.
	assign  irdy = !idone && (!ICap[$left(ICap)] || odone);
	// Emit while the output frame is unfinished and a full beat is ready -- or
	// the input is drained, flushing the residual and zero-padding the tail.
	assign  ovld = !odone && (full_beat || idone);
	for(genvar p = 0; p < PO0; p++)  assign  odat[p*GCD +: GCD] = Buf[p];

endmodule : dwc_generalized
