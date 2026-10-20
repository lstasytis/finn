#!/bin/bash
# usage: mvu_queue_run.sh <workdir> <tag> <axi.sv: orig|patched> [xvlog -d defines...] -- [plusargs...]
# compiles mvu_queue_tb.sv against finn-rtllib/mvu of build/wt/fix007 (or a patched mvu_vvu_axi.sv)
set -e
W=$1; tag=$2; axi=$3; shift 3
R=/home/lstasytis/finn/build/wt/fix007/finn-rtllib/mvu
T=$(dirname $(readlink -f $0))/mvu_queue_tb.sv
defs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do defs+=(-d "$1"); shift; done; shift || true
[ "$axi" = orig ] && A=$R/mvu_vvu_axi.sv || A=$axi
source /mnt/labstore/Xilinx/Vivado/2024.2/settings64.sh >/dev/null 2>&1
mkdir -p $W/$tag && cd $W/$tag
xvlog -sv "${defs[@]}" $R/mvu_pkg.sv $R/add_multi.sv $R/replay_buffer.sv $R/mvu.sv $R/mvu_vvu_8sx9_dsp58.sv $A $T > xvlog.log 2>&1 || { tail xvlog.log; exit 1; }
xvlog /mnt/labstore/Xilinx/Vivado/2024.2/data/verilog/src/glbl.v >> xvlog.log 2>&1
xelab --mt 4 --timescale 1ns/1ps -L unisims_ver -debug off mvu_queue_tb glbl -s tb > xelab.log 2>&1 || { tail xelab.log; exit 1; }
xsim tb -R "$@" > xsim.log 2>&1
grep -h "RESULT\|Overflowing" xsim.log | sort | uniq -c | sed "s/^/$tag: /" | head -5
