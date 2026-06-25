module dwc #(
	int unsigned  IBITS,
	int unsigned  OBITS
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

	if(IBITS == OBITS) begin : genNoop
		assign	irdy = ordy;
		assign	ovld = ivld;
		assign	odat  = idat;
	end : genNoop
	else if(IBITS < OBITS) begin : genUp

		// Sanity Checking: integer upscaling
		initial begin
			if(OBITS % IBITS) begin
				$error("Output width %0d is not a multiple of input width %0d.", OBITS, IBITS);
				$finish;
			end
		end

		// Parallelizing Shift Register A and Sidestep Buffer B on Input Path
		localparam int unsigned  K = OBITS / IBITS;
		typedef logic [IBITS-1:0]  dat_t;
		dat_t       [K-1:0]  ADat = 'x;
		logic [$clog2(K):0]  ACnt = K-1;	// (empty) K-1, ..., 0, -1 (full/valid)
		dat_t  BDat = 'x;
		logic  BRdy =  1;
		always_ff @(posedge clk) begin
			if(rst) begin
				ADat <= 'x;
				ACnt <= K-1;
				BDat <= 'x;
				BRdy <=  1;
			end
			else begin
				automatic type(ACnt)  acnt = (ovld && ordy)? K-1 : ACnt;
				automatic logic  rdy = !ovld || ordy;
				if((ivld || !BRdy) && rdy) begin
					ADat <= { BRdy? idat : BDat, ADat[K-1:1] };
					acnt--;
				end
				ACnt <= acnt;

				if(BRdy)  BDat <= idat;
				BRdy <= rdy || (BRdy && !ivld);
			end
		end

		// Output Assignments
		assign  irdy = BRdy;
		assign	ovld = ACnt[$left(ACnt)];
		assign	odat  = ADat;

	end : genUp
	else begin : genDown

		// Sanity Checking: integer downscaling
		initial begin
			if(IBITS % OBITS) begin
				$error("Input width %0d is not a multiple of output width %0d.", IBITS, OBITS);
				$finish;
			end
		end

		// Serializing Shift Register A and Sidestep Buffer B on Output Path
		localparam int unsigned  K = IBITS / OBITS;
		typedef logic [OBITS-1:0]  dat_t;
		dat_t [      K-1:0]  ADat = 'x;
		logic [$clog2(K):0]  ACnt =  1;	// (full) -K+1, ..., -1, 0, 1 (empty/not valid)
		dat_t  BDat = 'x;
		logic  BRdy =  1;
		dat_t  CDat = 'x;
		logic  CVld =  0;
		always_ff @(posedge clk) begin
			if(rst) begin
				ADat <= 'x;
				ACnt <=  1;
				BDat <= 'x;
				BRdy <=  1;
				CDat <= 'x;
				CVld <=  0;
			end
			else begin
				automatic type(ACnt)  acnt = ACnt;
				automatic logic       ainc = 0;
				if(irdy) begin
					ADat <= idat;
					acnt = ivld? -K+1 : 1;
				end
				else if(BRdy) begin
					ADat <= { {OBITS{1'bx}}, ADat[K-1:1] };
					ainc = BRdy;
				end;
				ACnt <= acnt + ainc;

				if(BRdy)  BDat <= ADat[0];
				BRdy <= !CVld || ordy || (BRdy && !ACnt[$left(ACnt)] && ACnt[0]);

				if(!CVld || ordy)  CDat <= BRdy? ADat[0] : BDat;
				CVld <= (CVld && !ordy) || !BRdy || ACnt[$left(ACnt)] || !ACnt[0];
			end
		end

		// Output Assignments
		assign  irdy = BRdy && !ACnt[$left(ACnt)];
		assign	ovld = CVld;
		assign	odat  = CDat;

	end : genDown

endmodule : dwc
