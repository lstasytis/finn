#!/bin/bash
# check.sh: build+run C-sim correctness, then synth a standard case set in parallel.
export PATH=/mnt/labstore/Xilinx/Vitis_HLS/2023.1/bin:$PATH
XILINX_HLS=/mnt/labstore/Xilinx/Vitis_HLS/2023.1
cd /home/lstasytis/backup/finn/claude-tools/dwc
echo "=== C-sim correctness ==="
g++ -std=c++14 -I$XILINX_HLS/include -I../../deps/finn-hlslib tb_dwc.cpp -o tb_dwc 2>&1 | head -5
./tb_dwc 2>/dev/null | tail -1
echo "=== synth sweep ==="
./sweep.sh "128 64" "64 128" "48 32" "96 40" "88 40" "44 48" "96 128" "128 96" "1024 768" "480 512" "1024 1536" "768 1024" "40 96" "160 40"
