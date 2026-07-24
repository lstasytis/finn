# OOC synth+P&R timing of dwc_generalized at 300MHz. tclargs: IBITS OBITS NUM_IN NUM_OUT
set IB [lindex $argv 0]; set OB [lindex $argv 1]
set NI [lindex $argv 2]; set NO [lindex $argv 3]
set part xc7z020clg400-1
read_verilog -sv /home/lstasytis/backup/finn/finn-rtllib/dwc/hdl/dwc_generalized.sv
synth_design -top dwc_generalized -part $part -mode out_of_context \
  -generic IBITS=$IB -generic OBITS=$OB -generic NUM_IN=$NI -generic NUM_OUT=$NO
create_clock -name clk -period 3.333 [get_ports clk]
set_input_delay  -clock clk 0.2 [all_inputs]
set_output_delay -clock clk 0.2 [all_outputs]
opt_design
place_design
phys_opt_design
route_design
phys_opt_design
set wns [get_property SLACK [get_timing_paths -delay_type max -nworst 1]]
set lut [llength [get_cells -hier -filter {PRIMITIVE_GROUP==LUT}]]
set ff  [llength [get_cells -hier -filter {PRIMITIVE_GROUP==FLOP_LATCH}]]
puts "RTLGEN ${IB}->${OB} (NI=$NI NO=$NO) WNS=${wns}ns LUT=${lut} FF=${ff}"
