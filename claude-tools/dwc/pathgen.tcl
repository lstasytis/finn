set part xc7z020clg400-1
read_verilog -sv /home/lstasytis/backup/finn/finn-rtllib/dwc/hdl/dwc_generalized.sv
synth_design -top dwc_generalized -part $part -mode out_of_context -generic IBITS=96 -generic OBITS=40 -generic NUM_IN=5 -generic NUM_OUT=12
create_clock -name clk -period 3.333 [get_ports clk]
set_input_delay -clock clk 0.2 [all_inputs]; set_output_delay -clock clk 0.2 [all_outputs]
opt_design; place_design; phys_opt_design; route_design
report_timing -delay_type max -max_paths 1 -nworst 1
