#!/bin/bash
# Save a TAV model (tav_model.py) of the newest fx/fifo, fx/fifo2 heuristic_analytical build of
# every model/board, with the fix013-int worktree. Output: $FINN_BUILD_DIR/tav/<key>.onnx
# usage: gen_tavs.sh  (writes its own PID to $FINN_BUILD_DIR/tav/gen.pid)
cd /home/lstasytis/finn/build/wt/fix013-int
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export NUM_DEFAULT_WORKERS=8
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/013-fifo-latency-model/scripts
O=$FINN_BUILD_DIR/tav
mkdir -p $O
echo $$ > $O/gen.pid
F=/home/lstasytis/finn/build/finn_build/fx
declare -A pick
for b in $(ls -dtr $F/fifo/fifo-heuristic_analytical_* $F/fifo2/fifo-heuristic_analytical_*); do
  key=$(basename $b | sed 's/^fifo-heuristic_analytical_//; s/_[^_]*$//')
  pick[$key]=$b   # newest wins (ls -tr)
done
for key in $(printf '%s\n' "${!pick[@]}" | sort); do
  [ -f $O/$key.onnx ] && continue
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  echo "$(date +%T) $key ${pick[$key]}" >> $O/gen.log
  ( cd $O && timeout 5400 python $S/tav_model.py ${pick[$key]} $O/$key.onnx > $O/$key.log 2>&1 )
  echo "$(date +%T) $key rc=$? $(tail -1 $O/$key.log)" >> $O/gen.log
done
echo DONE >> $O/gen.log
