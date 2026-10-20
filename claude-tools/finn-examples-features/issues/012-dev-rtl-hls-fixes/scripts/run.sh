#!/bin/bash
# usage: run.sh <tag> <rtl dir> [defines...] -- [plusargs...]
set -e
S=/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/i012
tag=$1; R=$2; shift 2
defs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do defs+=(-d "$1"); shift; done; shift || true
source /mnt/labstore/Xilinx/Vivado/2024.2/settings64.sh >/dev/null 2>&1
mkdir -p $S/runs/$tag && cd $S/runs/$tag
xvlog -sv "${defs[@]}" $R/mvu_pkg.sv $R/add_multi.sv $R/replay_buffer.sv $R/mvu.sv $R/mvu_vvu_8sx9_dsp58.sv $R/mvu_vvu_axi.sv $S/mvu_queue_tb.sv > xvlog.log 2>&1 || { tail xvlog.log; exit 1; }
xvlog /mnt/labstore/Xilinx/Vivado/2024.2/data/verilog/src/glbl.v >> xvlog.log 2>&1
xelab --mt 2 --timescale 1ns/1ps -L unisims_ver -debug off mvu_queue_tb glbl -s tb > xelab.log 2>&1 || { tail xelab.log; exit 1; }
xsim tb -R "$@" > xsim.log 2>&1
echo "$tag: $(grep -h RESULT xsim.log) asserts=$(grep -c Overflowing xsim.log)"
