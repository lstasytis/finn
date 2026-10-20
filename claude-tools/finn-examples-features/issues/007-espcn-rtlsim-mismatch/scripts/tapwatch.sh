#!/bin/bash
# usage: tapwatch.sh <run dir name> <interval s> <fifo names...> ; one summary line per interval
S=/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/s
run=$1; iv=$2; shift 2
cd /home/lstasytis/finn/build/wt/fix007 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
while true; do
  last=$(tail -n 1 $S/$run/$1.tap 2>/dev/null | cut -d' ' -f1)
  echo "$run: $1 last cycle ${last:-0}; $(python $S/tapcheck.py $S/$run/sub.onnx $S/$run "$@" 2>/dev/null | sed 's/StreamingFIFO_rtl_/F/g; s/ <- / </g; s/Thresholding/Thr/g' | tr '\n' '|')"
  if grep -q '^{\|Traceback' $S/$run.log; then echo "$run finished: $(grep '^{\|Traceback' $S/$run.log | tail -1)"; exit 0; fi
  sleep $iv
done
