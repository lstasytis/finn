#!/bin/bash
# hls_path.sh <gen|plain> INW OUTW  -> WNS + the worst post-route timing path.
set -e
KIND=$1; INW=$2; OUTW=$3
export FINN_ROOT=/home/lstasytis/backup/finn INW OUTW
REF=ref_gen; [ "$KIND" = plain ] && REF=ref_plain
cd /home/lstasytis/backup/finn/claude-tools/dwc
D=$(mktemp -d /tmp/hlspath.XXXXXX)
cp $REF/top.cpp $REF/run.tcl "$D"/
( cd "$D" && /mnt/labstore/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls -f run.tcl >hls.log 2>&1 )
VDIR="$D/proj/sol1/syn/verilog"
cat > "$D/t.tcl" <<TCL
set part xc7z020clg400-1
foreach f [glob -nocomplain $VDIR/*.v $VDIR/*.sv] { read_verilog \$f }
synth_design -top top -part \$part -mode out_of_context
create_clock -name ap_clk -period 3.333 [get_ports ap_clk]
set_input_delay  -clock ap_clk 0.2 [all_inputs]
set_output_delay -clock ap_clk 0.2 [all_outputs]
opt_design
place_design
phys_opt_design
route_design
phys_opt_design
puts "=== WNS ==="
puts [get_property SLACK [get_timing_paths -delay_type max -nworst 1]]
puts "=== WORST PATH ==="
report_timing -delay_type max -max_paths 1 -nworst 1 -input_pins
TCL
( cd "$D" && /mnt/labstore/Xilinx/Vivado/2023.1/bin/vivado -mode batch -source t.tcl >vivado.log 2>&1 )
sed -n '/=== WNS ===/,/Startpoint/p' "$D/vivado.log"
echo "--- path detail ---"
awk '/Slack/{p=1} p{print} /^\s*arrival time/{exit}' "$D/vivado.log" | head -60
rm -rf "$D"
