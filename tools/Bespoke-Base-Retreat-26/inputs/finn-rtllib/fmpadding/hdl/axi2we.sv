module axi2we #(
	int unsigned  ADDR_BITS
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Lite ------------------------
	// Writing
	input	                 s_axilite_AWVALID,
	output	                 s_axilite_AWREADY,
	input	[ADDR_BITS-1:0]  s_axilite_AWADDR,

	input	        s_axilite_WVALID,
	output	        s_axilite_WREADY,
	input	[31:0]  s_axilite_WDATA,
	input	[ 3:0]  s_axilite_WSTRB,

	output	       s_axilite_BVALID,
	input	       s_axilite_BREADY,
	output	[1:0]  s_axilite_BRESP,

	// Reading tied to all-ones
	input	       s_axilite_ARVALID,
	output	       s_axilite_ARREADY,
	input	[ADDR_BITS-1:0]  s_axilite_ARADDR,

	output	        s_axilite_RVALID,
	input	        s_axilite_RREADY,
	output	[31:0]  s_axilite_RDATA,
	output	[ 1:0]  s_axilite_RRESP,

	// Write Enable Interface
	output	logic                  we,
	output	logic [ADDR_BITS-1:0]  wa,
	output	logic [         31:0]  wd
);

	uwire  clk = ap_clk;
	uwire  rst = !ap_rst_n;


	logic  WABusy = 0;
	logic  WDBusy = 0;
	logic [ADDR_BITS-1:0]  Addr = 'x;
	logic [         31:0]  Data = 'x;

	assign	we = WABusy && WDBusy && s_axilite_BREADY;
	assign	wa = Addr;
	assign	wd = Data;

	uwire  clr_wr = rst || we;
	always_ff @(posedge clk) begin
		if(clr_wr) begin
			WABusy <= 0;
			Addr <= 'x;
			WDBusy <= 0;
			Data <= 'x;
		end
		else begin
			if(!WABusy) begin
				WABusy <= s_axilite_AWVALID;
				Addr   <= s_axilite_AWADDR;
			end
			if(!WDBusy) begin
				WDBusy <= s_axilite_WVALID;
				Data   <= s_axilite_WDATA;
			end
		end
	end
	assign	s_axilite_AWREADY = !WABusy;
	assign	s_axilite_WREADY  = !WDBusy;
	assign	s_axilite_BVALID  = WABusy && WDBusy;
	assign	s_axilite_BRESP   = '0; // OK

	// Answer all reads with '1
	logic  RValid =  0;
	uwire  clr_rd = rst || (RValid && s_axilite_RREADY);
	always_ff @(posedge clk) begin
		if(clr_rd)        RValid <=  0;
		else if(!RValid)  RValid <= s_axilite_ARVALID;
	end
	assign	s_axilite_ARREADY = !RValid;
	assign	s_axilite_RVALID  = RValid;
	assign	s_axilite_RDATA   = '1;
	assign	s_axilite_RRESP   = '0; // OK

endmodule : axi2we
