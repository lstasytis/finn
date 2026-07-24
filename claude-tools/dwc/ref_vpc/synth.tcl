# OOC synth of the vpc (Vector Pack Converter) RTL, mirroring ref_rtl/synth.tcl.
# Params: W (element bits), N (elems/vector), PI (in lanes), PO (out lanes).
# Bit-width mapping used by the benchmark: W=1, PI=inWidth, PO=outWidth,
# N=lcm(PI,PO) (one exact repacking frame). vpc normalizes by gcd(PI,PO), so
# W=1 yields the same minimal normalized datapath as W=gcd(inW,outW).
read_verilog -sv [list $::env(VPC_SRC)]
synth_design -top vpc -mode out_of_context -part xc7z020clg400-1 \
  -generic W=$::env(W) -generic N=$::env(N) -generic PI=$::env(PI) -generic PO=$::env(PO)
set rpt [report_utilization -return_string]
set luts 0; set ffs 0
regexp {Slice LUTs\*?\s*\|\s*([0-9]+)} $rpt -> luts
if {$luts == 0} { regexp {CLB LUTs\*?\s*\|\s*([0-9]+)} $rpt -> luts }
regexp {Slice Registers\s*\|\s*([0-9]+)} $rpt -> ffs
if {$ffs == 0} { regexp {CLB Registers\s*\|\s*([0-9]+)} $rpt -> ffs }
puts "VPC_RESULT LUT=$luts FF=$ffs"
exit
