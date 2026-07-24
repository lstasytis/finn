#!/bin/bash
cd /home/lstasytis/backup/finn/claude-tools/dwc
O=/tmp/texmeas
for p in "25 10" "105 10" "305 10" "505 10" "755 10" "1005 10" "10 25" "10 105" "10 305" "10 505" "10 755" "10 1005" "44 48" "88 96" "176 192" "352 384" "420 448" "480 512" "720 768"; do
  set -- $p
  ( ./measure_point.sh $1 $2 > "$O/${1}_${2}.txt" 2>/dev/null ) &
  while [ $(jobs -r|wc -l) -ge 10 ]; do wait -n; done
done
wait
echo MEAS_DONE > $O/status
