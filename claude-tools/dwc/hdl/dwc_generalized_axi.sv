/******************************************************************************
 * Copyright (C) 2026, Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	AXI-Stream adapter for the generalized (non-multiple + padding) DWC.
 *****************************************************************************/
module dwc_generalized_axi #(
	int unsigned  IBITS,
	int unsigned  OBITS,
	int unsigned  NUM_IN,
	int unsigned  NUM_OUT,

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

	dwc_generalized #(.IBITS(IBITS), .OBITS(OBITS), .NUM_IN(NUM_IN), .NUM_OUT(NUM_OUT)) core (
		.clk(ap_clk), .rst(!ap_rst_n),
		.irdy(s_axis_tready), .ivld(s_axis_tvalid), .idat(s_axis_tdata[IBITS-1:0]),
		.ordy(m_axis_tready), .ovld(m_axis_tvalid), .odat(m_axis_tdata[OBITS-1:0])
	);
	if(OBITS < AXI_OBITS) begin
		assign	m_axis_tdata[AXI_OBITS-1:OBITS] = '0;
	end

endmodule : dwc_generalized_axi
