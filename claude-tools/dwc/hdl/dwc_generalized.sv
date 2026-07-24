/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Generalized Stream Data Width Converter (non-multiple + padding),
 *		pipelined for 300 MHz at II=1.
 *
 * @description
 *  Converts NUM_IN input beats (IBITS) per frame into NUM_OUT output beats
 *  (OBITS), preserving the flat little-endian bit sequence. Handles ARBITRARY
 *  (non-multiple/coprime) ratios and, when NUM_IN*IBITS != NUM_OUT*OBITS,
 *  zero-PADDING (output longer) or CROPPING (output shorter). Frames stream
 *  back-to-back. GCD-normalized to a lane granularity (W0=GCD bits).
 *
 *  Timing: the data-dependent placement (a log-depth barrel shifting the input
 *  word to bit offset off*W0) is REGISTERED into a 1-entry prefetch slot
 *  (Placed), so it is out of the accumulate recurrence. The accumulate step is
 *  only a constant-shift emit + OR, a short path. The placement offset off_k =
 *  (k*PI0) mod PO0 is a pure function of the input index (independent of buffer
 *  contents / backpressure), so the pre-shifted word is merged into Buf exactly
 *  when the occupancy has drained to off_k -- keeping the two in lock-step.
 *  irdy/ovld depend only on registers (no combinational ordy->irdy path).
 *****************************************************************************/
module dwc_generalized #(
	int unsigned  IBITS,
	int unsigned  OBITS,
	int unsigned  NUM_IN,
	int unsigned  NUM_OUT
)(
	input	logic  clk,
	input	logic  rst,

	output	logic  irdy,
	input	logic  ivld,
	input	logic [IBITS-1:0]  idat,

	input	logic  ordy,
	output	logic  ovld,
	output	logic [OBITS-1:0]  odat
);
	function automatic int unsigned gcd(input int unsigned a, input int unsigned b);
		return (b == 0)? a : gcd(b, a % b);
	endfunction
	localparam int unsigned  GCD = gcd(IBITS, OBITS);
	localparam int unsigned  W0  = GCD;
	localparam int unsigned  PI0 = IBITS / GCD;          // input lanes / beat
	localparam int unsigned  PO0 = OBITS / GCD;          // output lanes / beat
	localparam int unsigned  BUFLANES = PI0 + PO0;
	localparam int unsigned  BUFW = IBITS + OBITS - GCD; // max meaningful buffer width
	// cropping (input carries more bits than the output frame) can leave residue
	localparam bit  CROP = (NUM_IN*IBITS > NUM_OUT*OBITS);

	// Frame progress (count down: N-1 .. 0 .. -1 = done)
	logic signed [$clog2(NUM_IN)+1:0]   ITrn = NUM_IN-1;
	logic signed [$clog2(NUM_OUT)+1:0]  OTrn = NUM_OUT-1;
	uwire  idone = ITrn[$left(ITrn)];
	uwire  odone = OTrn[$left(OTrn)];

	// Accumulate buffer + readiness. Rdy = (valid lanes) - PO0, so a full output
	// beat is ready exactly when Rdy >= 0 (a single sign-bit test, not a
	// magnitude compare -- keeps it off the critical path).
	logic [BUFW-1:0]                 Buf = '0;
	logic signed [$clog2(BUFLANES+1)+1:0]  Rdy = -PO0;   // -PO0 .. PI0

	// Prefetch slot: input word already barrel-shifted to its placement offset
	logic [BUFW-1:0]                 Placed = '0;
	logic                            PlacedVld = 1'b0;
	logic [$clog2(PO0+1):0]          off = '0;          // 0 .. PO0-1

	// Combinational control (register-only -> no ordy->irdy path)
	uwire  full_beat = !Rdy[$left(Rdy)];                // Rdy >= 0
	uwire  flush     = idone && !PlacedVld;             // input drained: pad/residual
	uwire  can_merge = PlacedVld && Rdy[$left(Rdy)];    // Rdy < 0 -> room for the word
	assign ovld = !odone && (full_beat || flush);
	// Accept while the input frame is unfinished and either the output frame is
	// already done (drain/crop) or the prefetch slot is (about to be) free.
	assign irdy = !idone && (odone || !PlacedVld || can_merge);
	assign odat = Buf[OBITS-1:0];

	uwire  do_emit   = ovld && ordy;
	uwire  do_accept = ivld && irdy;
	uwire  do_store  = do_accept && !odone;            // barrel & keep (else crop-drain)
	uwire  do_merge  = can_merge;

	// next-offset: (off + PI0) mod PO0  (constant modulus -> synthesizable)
	function automatic logic [$clog2(PO0+1):0] nextoff(input logic [$clog2(PO0+1):0] o);
		return ((o + PI0) % PO0);
	endfunction

	always_ff @(posedge clk) begin
		if(rst) begin
			Buf <= '0; Rdy <= -PO0;
			Placed <= '0; PlacedVld <= 1'b0; off <= '0;
			ITrn <= NUM_IN-1; OTrn <= NUM_OUT-1;
		end
		else begin
			automatic logic [BUFW-1:0]  n_buf = Buf;
			automatic logic signed [$clog2(BUFLANES+1)+1:0]  n_rdy = Rdy;
			automatic logic signed [$clog2(NUM_IN)+1:0]   n_itrn = ITrn;
			automatic logic signed [$clog2(NUM_OUT)+1:0]  n_otrn = OTrn;

			// Emit (constant-shift): vacated top lanes become zero (padding).
			if(do_emit) begin
				n_buf = n_buf >> OBITS;
				n_rdy = full_beat? (n_rdy - PO0) : -PO0;   // flush empties the buffer
				n_otrn--;
			end
			// Merge the pending pre-shifted word (aligned: Rdy has drained to off-PO0).
			if(do_merge) begin
				n_buf = n_buf | Placed;
				n_rdy = n_rdy + PI0;
			end
			Buf <= n_buf;

			// Prefetch slot / offset update
			if(do_store) begin
				Placed    <= BUFW'(idat) << (off * W0);
				PlacedVld <= 1'b1;
				off       <= nextoff(off);
			end
			else if(do_merge) begin
				PlacedVld <= 1'b0;
			end
			if(do_accept)  n_itrn--;

			// Frame restart when both sides complete.
			if(n_itrn[$left(n_itrn)] && n_otrn[$left(n_otrn)]) begin
				n_itrn = NUM_IN-1;
				n_otrn = NUM_OUT-1;
				off  <= '0;
				n_rdy = -PO0;
				PlacedVld <= 1'b0;     // discard any pending (cropped) prefetch
				if(CROP)  Buf <= '0;   // discard cropped residue
			end
			ITrn <= n_itrn;
			OTrn <= n_otrn;
			Rdy  <= n_rdy;
		end
	end

endmodule : dwc_generalized
