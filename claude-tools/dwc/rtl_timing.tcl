# OOC synth + timing of the RTL dwc at 300MHz. Args via -tclargs IBITS OBITS
set IBITS [lindex $argv 0]
set OBITS [lindex $argv 1]
set part xc7z020clg400-1
read_verilog -sv /home/lstasytis/backup/finn/finn-rtllib/dwc/hdl/dwc.sv
synth_design -top dwc -part $part -mode out_of_context \
  -generic IBITS=$IBITS -generic OBITS=$OBITS
create_clock -name clk -period 3.333 [get_ports clk]
# reasonable IO delays so the path is core-dominated
set_input_delay  -clock clk 0.2 [all_inputs]
set_output_delay -clock clk 0.2 [all_outputs]
opt_design
place_design
route_design
set wns [get_property SLACK [get_timing_paths -delay_type max -nworst 1]]
set lut [llength [get_cells -hier -filter {PRIMITIVE_GROUP==LUT}]]
set ff  [llength [get_cells -hier -filter {PRIMITIVE_GROUP==FLOP_LATCH}]]
puts "RTL_RESULT ${IBITS}->${OBITS} WNS=${wns}ns LUT=${lut} FF=${ff}"
