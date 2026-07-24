#!/bin/bash
# pr_sweep.sh <gen|plain> "IN OUT" ...  -> real Vivado P&R WNS/LUT/FF per pair, parallel.
KIND=$1; shift
cd /home/lstasytis/backup/finn/claude-tools/dwc
OUT=$(mktemp -d)
for p in "$@"; do
  set -- $p
  ( ./hls_timing.sh $KIND $1 $2 > "$OUT/${1}_${2}.txt" 2>/dev/null ) &
  while [ $(jobs -r | wc -l) -ge 12 ]; do wait -n; done
done
wait
for p in "$@"; do set -- $p; cat "$OUT/${1}_${2}.txt" 2>/dev/null; done
rm -rf "$OUT"
