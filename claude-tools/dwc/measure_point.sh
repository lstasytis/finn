#!/bin/bash
# measure_point.sh IN OUT  -> writes newrtl and newhls LUT/FF for one width pair.
export PATH=/mnt/labstore/Xilinx/Vitis_HLS/2023.1/bin:/mnt/labstore/Xilinx/Vivado/2023.1/bin:$PATH
IN=$1; OUT=$2
read NI NO < <(python3 -c "import math;L=math.lcm($IN,$OUT);print(L//$IN, L//$OUT)")
# --- new RTL ---
D=$(mktemp -d)
( cd $D && vivado -mode batch -source /home/lstasytis/backup/finn/claude-tools/dwc/rtlgen_timing.tcl -tclargs $IN $OUT $NI $NO >v.log 2>&1 )
R=$(grep "RTLGEN" $D/v.log | grep -v "puts " | head -1)
rlut=$(echo "$R" | grep -oE "LUT=[0-9]+" | cut -d= -f2)
rff=$(echo "$R" | grep -oE "FF=[0-9]+" | cut -d= -f2)
rm -rf $D
# --- new HLS (hoisted streamtools.h, real OOC) ---
H=$(/home/lstasytis/backup/finn/claude-tools/dwc/hls_timing.sh gen $IN $OUT 2>/dev/null)
hlut=$(echo "$H" | grep -oE "LUT=[0-9]+" | cut -d= -f2)
hff=$(echo "$H" | grep -oE "FF=[0-9]+" | cut -d= -f2)
echo "PT $IN $OUT newrtl ${rlut:-NA} ${rff:-NA} newhls ${hlut:-NA} ${hff:-NA}"
