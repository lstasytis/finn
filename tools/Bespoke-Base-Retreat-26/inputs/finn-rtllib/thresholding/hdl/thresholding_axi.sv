module thresholding_axi #(
	int unsigned  WI,	// input precision
	int unsigned  WT,	// threshold precision
	int unsigned  N,	// number of thresholds
	int unsigned  C = 1,	// Channels
	int unsigned  PE = 1,	// Processing Parallelism, requires C = k*PE

	bit  SIGNED = 1,	// signed inputs
	bit  FPARG  = 0,	// floating-point inputs: [sign] | exponent | mantissa
	int  BIAS   = 0,	// offsetting the output [0, 2^N-1] -> [BIAS, 2^N-1 + BIAS]

	int unsigned  SETS = 1,  // Number of independent threshold sets

	// Initial Thresholds
	parameter  THRESHOLDS_PATH = "",
	bit  USE_AXILITE,	// Implement AXI-Lite for threshold read/write

	// Force Use of On-Chip Memory Blocks
	int unsigned  DEPTH_TRIGGER_URAM = 0,	// if non-zero, local mems of this depth or more go into URAM (prio)
	int unsigned  DEPTH_TRIGGER_BRAM = 0,	// if non-zero, local mems of this depth or more go into BRAM
	bit  DEEP_PIPELINE = 0,

	localparam int unsigned  CF = C/PE,	// Channel Fold
	localparam int unsigned  IP_ADDR_WIDTH = $clog2(CF) + $clog2(PE) + $clog2(N),
	localparam int unsigned  ADDR_FOLD = 1 + (WT-1)/32,
	localparam int unsigned  ADDR_WIDTH = IP_ADDR_WIDTH + $clog2(ADDR_FOLD) + 2,
	localparam int unsigned  S_BITS = SETS > 2? $clog2(SETS) : 1,
	localparam int unsigned  O_BITS = BIAS >= 0?
		/* unsigned */ $clog2(N+BIAS+1) :
		/* signed */ 1+$clog2(-BIAS >= N+BIAS+1? -BIAS : N+BIAS+1)
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	logic                   s_axilite_AWVALID,
	output	logic                   s_axilite_AWREADY,
	input	logic [ADDR_WIDTH-1:0]  s_axilite_AWADDR,	// lowest 2 bits (byte selectors) are ignored

	input	logic         s_axilite_WVALID,
	output	logic         s_axilite_WREADY,
	input	logic [31:0]  s_axilite_WDATA,
	input	logic [ 3:0]  s_axilite_WSTRB,

	output	logic        s_axilite_BVALID,
	input	logic        s_axilite_BREADY,
	output	logic [1:0]  s_axilite_BRESP,

	// Reading
	input	logic                   s_axilite_ARVALID,
	output	logic                   s_axilite_ARREADY,
	input	logic [ADDR_WIDTH-1:0]  s_axilite_ARADDR,

	output	logic         s_axilite_RVALID,
	input	logic         s_axilite_RREADY,
	output	logic [31:0]  s_axilite_RDATA,
	output	logic [ 1:0]  s_axilite_RRESP,

	//- AXI Stream - Set Selection ------
	// (ignored for SETS < 2)
	output	logic  s_axis_set_tready,
	input	logic  s_axis_set_tvalid,
	input	logic [((S_BITS+7)/8)*8-1:0]  s_axis_set_tdata,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [((PE*WI+7)/8)*8-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [((PE*O_BITS+7)/8)*8-1:0]  m_axis_tdata
);

	//-----------------------------------------------------------------------
	// AXI-lite Configuration Interface
	uwire  cfg_en;
	uwire  cfg_we;
	uwire [IP_ADDR_WIDTH-1:0]  cfg_a;
	uwire [WT-1:0]  cfg_d;
	uwire  cfg_rack;
	uwire [WT-1:0]  cfg_q;

	if(USE_AXILITE) begin
		axilite #(.ADDR_WIDTH(ADDR_WIDTH), .DATA_WIDTH(32), .IP_DATA_WIDTH(WT)) axi (
			.aclk(ap_clk), .aresetn(ap_rst_n),

			.awready(s_axilite_AWREADY), .awvalid(s_axilite_AWVALID), .awaddr(s_axilite_AWADDR), .awprot('x),
			.wready(s_axilite_WREADY),   .wvalid(s_axilite_WVALID),   .wdata(s_axilite_WDATA),   .wstrb(s_axilite_WSTRB),
			.bready(s_axilite_BREADY),   .bvalid(s_axilite_BVALID),   .bresp(s_axilite_BRESP),

			.arready(s_axilite_ARREADY), .arvalid(s_axilite_ARVALID), .araddr(s_axilite_ARADDR), .arprot('x),
			.rready(s_axilite_RREADY),   .rvalid(s_axilite_RVALID),   .rresp(s_axilite_RRESP),   .rdata(s_axilite_RDATA),

			.ip_en(cfg_en), .ip_wen(cfg_we), .ip_addr(cfg_a), .ip_wdata(cfg_d),
			.ip_rack(cfg_rack), .ip_rdata(cfg_q)
		);
	end
	else begin
		assign	cfg_en =  0;
		assign	cfg_we = 'x;
		assign	cfg_a  = 'x;
		assign	cfg_d  = 'x;
	end

	//-----------------------------------------------------------------------
	// Join set selection and data streams
	uwire  irdy;
	uwire  ivld;
	uwire [S_BITS-1:0]  iset;
	if(SETS < 2) begin
		assign	ivld = s_axis_tvalid;
		assign	iset = 'x;

		assign  s_axis_tready     = irdy;
		assign  s_axis_set_tready = 'x;
	end
	else begin
		assign	ivld = s_axis_tvalid && s_axis_set_tvalid;
		assign	iset = s_axis_set_tdata[S_BITS-1:0];

		assign  s_axis_tready     = irdy && s_axis_set_tvalid;
		assign  s_axis_set_tready = irdy && s_axis_tvalid;
	end

	//-----------------------------------------------------------------------
	// Cast Inputs into Threshold Data Type
	uwire [PE-1:0][WT-1:0]  idat;
	for(genvar  pe = 0; pe < PE; pe++) begin
		if(WT == WI) begin : genCopy
			assign	idat[pe] = s_axis_tdata[pe*WI+:WI];
		end : genCopy
		else begin
			initial begin
				if(FPARG) begin
					$error("%m: Can't cast floating-point type.");
					$finish;
				end
			end

			if(WT > WI) begin : genWiden
				assign	idat[pe] = { {(WT-WI){SIGNED? s_axis_tdata[(pe+1)*WI-1] : 1'b0}}, s_axis_tdata[pe*WI+:WI] };
			end : genWiden
			else begin : genNarrow
				// Saturate for clipping inputs
				if(!SIGNED) begin
					assign	idat[pe] = |s_axis_tdata[pe*WI+WT+:WI-WT]? '1 : s_axis_tdata[pe*WI+:WT];
				end
				else begin
					assign	idat[pe] =
						(s_axis_tdata[pe*WI+WT+:WI-WT] == {(WI-WT){s_axis_tdata[pe*WI+WT-1]}})? s_axis_tdata[pe*WI+:WT] :
						{s_axis_tdata[(pe+1)*WI-1], {(WT-1){!s_axis_tdata[(pe+1)*WI-1]}}};
				end
			end : genNarrow
		end
	end

	//-----------------------------------------------------------------------
	// Kernel Implementation
	thresholding #(
		.K(WT), .N(N), .C(C), .PE(PE),
		.SIGNED(SIGNED), .FPARG(FPARG), .BIAS(BIAS),
		.SETS(SETS),
		.THRESHOLDS_PATH(THRESHOLDS_PATH), .USE_CONFIG(USE_AXILITE),
		.DEPTH_TRIGGER_URAM(DEPTH_TRIGGER_URAM), .DEPTH_TRIGGER_BRAM(DEPTH_TRIGGER_BRAM),
		.DEEP_PIPELINE(DEEP_PIPELINE)
	) impl (
		.clk(ap_clk), .rst(!ap_rst_n),

		.cfg_en, .cfg_we, .cfg_a, .cfg_d,
		.cfg_rack, .cfg_q,

		.irdy, .ivld, .iset, .idat,
		.ordy(m_axis_tready), .ovld(m_axis_tvalid), .odat(m_axis_tdata[PE*O_BITS-1:0])
	);
	if($bits(m_axis_tdata) > PE*O_BITS) begin : genPadOut
		assign	m_axis_tdata[$left(m_axis_tdata):PE*O_BITS] = '0;
	end : genPadOut

endmodule : thresholding_axi
