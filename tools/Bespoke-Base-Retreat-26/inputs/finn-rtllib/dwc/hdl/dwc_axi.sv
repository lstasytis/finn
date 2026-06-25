module dwc_axi #(
	int unsigned  IBITS,
	int unsigned  OBITS,

	localparam int unsigned  AXI_IBITS = (IBITS+7)/8 * 8,
	localparam int unsigned  AXI_OBITS = (OBITS+7)/8 * 8
)(
	//- Global Control ------------------
	input	logic  ap_clk,
	input	logic  ap_rst_n,

	//- AXI Stream - Input --------------
	output	logic  s_axis_tready,
	input	logic  s_axis_tvalid,
	input	logic [AXI_IBITS-1:0]  s_axis_tdata,

	//- AXI Stream - Output -------------
	input	logic  m_axis_tready,
	output	logic  m_axis_tvalid,
	output	logic [AXI_OBITS-1:0]  m_axis_tdata
);

	dwc #(.IBITS(IBITS), .OBITS(OBITS)) core (
		.clk(ap_clk), .rst(!ap_rst_n),
		.irdy(s_axis_tready), .ivld(s_axis_tvalid), .idat(s_axis_tdata[IBITS-1:0]),
		.ordy(m_axis_tready), .ovld(m_axis_tvalid), .odat(m_axis_tdata[OBITS-1:0])
	);
	if(OBITS < AXI_OBITS) begin
		assign	m_axis_tdata[AXI_OBITS-1:OBITS] = '0;
	end

endmodule : dwc_axi
