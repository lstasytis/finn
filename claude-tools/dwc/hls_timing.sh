#!/bin/bash
# hls_timing.sh <gen|plain> INW OUTW
# 1) vitis_hls csynth the kernel (emits Verilog), 2) real Vivado OOC synth+P&R
# at 300MHz on the generated RTL, report WNS/LUT/FF. This is the REAL timing.
set -e
KIND=$1; INW=$2; OUTW=$3
export FINN_ROOT=/home/lstasytis/backup/finn INW OUTW
REF=ref_gen; [ "$KIND" = plain ] && REF=ref_plain
cd /home/lstasytis/backup/finn/claude-tools/dwc
D=$(mktemp -d /tmp/hlstim.XXXXXX)
cp $REF/top.cpp $REF/run.tcl "$D"/
( cd "$D" && /mnt/labstore/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls -f run.tcl >hls.log 2>&1 ) || { echo "$INW->$OUTW HLS_FAIL"; rm -rf "$D"; exit 1; }
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
set wns [get_property SLACK [get_timing_paths -delay_type max -nworst 1]]
set lut [llength [get_cells -hier -filter {PRIMITIVE_GROUP==LUT}]]
set ff  [llength [get_cells -hier -filter {PRIMITIVE_GROUP==FLOP_LATCH}]]
puts "HLS_${KIND} ${INW}->${OUTW} WNS=\${wns}ns LUT=\${lut} FF=\${ff}"
TCL
( cd "$D" && /mnt/labstore/Xilinx/Vivado/2023.1/bin/vivado -mode batch -source t.tcl >vivado.log 2>&1 ) || { echo "$INW->$OUTW VIVADO_FAIL"; tail -20 "$D/vivado.log"; rm -rf "$D"; exit 1; }
grep -E "HLS_${KIND}" "$D/vivado.log" | tail -1
rm -rf "$D"
