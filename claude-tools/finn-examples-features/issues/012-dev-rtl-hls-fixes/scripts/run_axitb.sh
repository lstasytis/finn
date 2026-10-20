#!/bin/bash
# usage: run_axitb.sh <tag> <rtl dir> <tb file>
set -e
S=/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/i012
tag=$1; R=$2; T=$3
source /mnt/labstore/Xilinx/Vivado/2024.2/settings64.sh >/dev/null 2>&1
mkdir -p $S/runs/$tag && cd $S/runs/$tag
xvlog -sv $R/mvu_pkg.sv $R/add_multi.sv $R/replay_buffer.sv $R/mvu.sv $R/mvu_vvu_8sx9_dsp58.sv $R/mvu_vvu_axi.sv $T > xvlog.log 2>&1 || { tail xvlog.log; exit 1; }
xvlog /mnt/labstore/Xilinx/Vivado/2024.2/data/verilog/src/glbl.v >> xvlog.log 2>&1
xelab --mt 2 --timescale 1ns/1ps -L unisims_ver -debug off mvu_axi_tb glbl -s tb > xelab.log 2>&1 || { tail xelab.log; exit 1; }
timeout 1800 xsim tb -R > xsim.log 2>&1 || true
echo "$tag: $(grep -c 'Successfully' xsim.log) ok; errors: $(grep -E 'Error|mismatch|Missing|Spurious' xsim.log | head -3 | tr '\n' '|')"
