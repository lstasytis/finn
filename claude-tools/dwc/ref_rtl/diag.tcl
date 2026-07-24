read_verilog -sv [list $::env(FINN_ROOT)/finn-rtllib/dwc/hdl/dwc.sv $::env(FINN_ROOT)/finn-rtllib/dwc/hdl/dwc_axi.sv]
synth_design -top dwc_axi -mode out_of_context -part xc7z020clg400-1 -generic IBITS=1024 -generic OBITS=512
puts "===FULL UTIL==="
puts [report_utilization -return_string]
exit
