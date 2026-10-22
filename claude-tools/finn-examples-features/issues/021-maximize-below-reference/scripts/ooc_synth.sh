#!/bin/bash
# Out-of-context synthesis of a finished build's stitched IP (sources from
# stitched_ip/all_verilog_srcs.txt), per-instance utilization.
#   ooc_synth.sh <build dir> <part> <out dir> [threads]
# Writes <out>/util.rpt, <out>/util_hier.rpt, <out>/vivado.log; non-project flow, clock CLK_NS (5).
# PNR=1 also places and routes (OOC) and writes <out>/util_routed.rpt, <out>/timing_routed.rpt.
# FLATTEN sets synth_design -flatten_hierarchy (default rebuilt).
set -e
B=$(realpath "$1"); PART=$2; OUT=$(realpath -m "$3"); T=${4:-2}
mkdir -p "$OUT"
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/env24.sh
awk '!seen[$0]++' "$B/stitched_ip/all_verilog_srcs.txt" > "$OUT/srcs.txt"
python3 $(dirname $(realpath $0))/stub_xip.py "$OUT/srcs.txt" "$OUT/stubs"
FLATTEN=${FLATTEN:-rebuilt}
cat > "$OUT/synth.tcl" <<EOF
set_param general.maxThreads $T
set fp [open $OUT/srcs.txt r]
while {[gets \$fp f] >= 0} {
  if {\$f eq ""} continue
  if {[string match *.sv \$f]} { read_verilog -sv \$f } elseif {[string match *.vhd \$f]} { read_vhdl \$f } else { read_verilog \$f }
}
close \$fp
synth_design -top finn_design_wrapper -part $PART -mode out_of_context -flatten_hierarchy ${FLATTEN:-rebuilt}
create_clock -period ${CLK_NS:-5.0} [get_ports ap_clk]
report_utilization -file $OUT/util.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $OUT/util_hier.rpt
EOF
if [ "${PNR:-0}" = 1 ]; then cat >> "$OUT/synth.tcl" <<EOF
opt_design
place_design
route_design
report_utilization -file $OUT/util_routed.rpt
report_timing_summary -file $OUT/timing_routed.rpt
report_route_status -file $OUT/route_status.rpt
EOF
fi
cd "$OUT" && vivado -mode batch -nojournal -log "$OUT/vivado.log" -source "$OUT/synth.tcl" > /dev/null 2>&1
echo "done $OUT"
