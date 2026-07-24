set ibits $::env(IBITS)
set obits $::env(OBITS)
read_verilog -sv [list $::env(FINN_ROOT)/finn-rtllib/dwc/hdl/dwc.sv $::env(FINN_ROOT)/finn-rtllib/dwc/hdl/dwc_axi.sv]
synth_design -top dwc_axi -mode out_of_context -part xc7z020clg400-1 -generic IBITS=$ibits -generic OBITS=$obits
set rpt [report_utilization -return_string]
regexp {Slice LUTs[^\n]*?\|\s*([0-9]+)} $rpt -> luts
if {![info exists luts]} { regexp {CLB LUTs[^\n]*?\|\s*([0-9]+)} $rpt -> luts }
puts "RTL_LUT=$luts"
exit
