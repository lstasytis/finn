#!/bin/bash
# Run a set of INW,OUTW pairs (args: "IN OUT" ...) in parallel, collect results sorted.
export PATH=/mnt/labstore/Xilinx/Vitis_HLS/2023.1/bin:$PATH
cd /home/lstasytis/backup/finn/claude-tools/dwc
PAIRS=("$@")
OUT=$(mktemp -d)
pids=()
for p in "${PAIRS[@]}"; do
  set -- $p
  ( ./synth_one.sh $1 $2 > "$OUT/$1_$2.txt" 2>/dev/null ) &
  pids+=($!)
  # cap concurrency at 6
  while [ $(jobs -r | wc -l) -ge 14 ]; do wait -n; done
done
wait
for p in "${PAIRS[@]}"; do
  set -- $p
  cat "$OUT/$1_$2.txt" 2>/dev/null
done
rm -rf "$OUT"
