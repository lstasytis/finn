#!/bin/bash
# sweep_pr.sh <tag>  -- run a fixed case set through real P&R, collect to /tmp/<tag>/.
# Reads pairs from stdin ("IN OUT" per line). Avoids the set-- collect bug.
TAG=$1
cd /home/lstasytis/backup/finn/claude-tools/dwc
O=/tmp/$TAG; rm -rf $O; mkdir -p $O
pids=""
while read -r a b; do
  [ -z "$a" ] && continue
  ( ./hls_timing.sh gen $a $b > "$O/${a}_${b}.txt" 2>/dev/null ) &
  while [ $(jobs -r | wc -l) -ge 10 ]; do wait -n; done
done
wait
cat $O/*.txt
echo "SWEEP_DONE"
