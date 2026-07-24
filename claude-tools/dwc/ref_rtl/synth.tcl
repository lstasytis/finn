set top $::env(TOPMOD)
read_verilog -sv [list $::env(FINN_ROOT)/finn-rtllib/dwc/hdl/dwc.sv $::env(FINN_ROOT)/finn-rtllib/dwc/hdl/dwc_axi.sv]
synth_design -top $top -mode out_of_context -part xc7z020clg400-1 -generic IBITS=$::env(IBITS) -generic OBITS=$::env(OBITS)
report_utilization -file util.rpt
exit
