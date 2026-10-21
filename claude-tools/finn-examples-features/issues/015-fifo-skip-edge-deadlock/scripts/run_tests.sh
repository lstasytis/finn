#!/bin/bash
# Issue 015: run transformer feature tests one after the other in a worktree.
# usage: run_tests.sh <worktree> <lane> "<pytest -k expr>" ...   (PID in $FINN_BUILD_DIR/logs/<lane>.pid)
W=$1; LANE=$2; shift 2
cd $W
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export NUM_DEFAULT_WORKERS=4
L=$FINN_BUILD_DIR/logs; mkdir -p $L
echo $$ > $L/$LANE.pid
T=${TFILE:-tests/benchmark/transformer/test_build_transformer.py}
for k in "$@"; do
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  tag=$(echo $k | tr ' []' '___')
  echo "$(date +%T) start $k" >> $L/$LANE.log
  timeout ${TIMEOUT:-7200} python -m pytest -p no:cacheprovider -v $T -k "$k" \
    --junitxml=$L/${LANE}_$tag.xml > $L/${LANE}_$tag.log 2>&1
  echo "$(date +%T) rc=$? $k $(grep -E 'PASSED|FAILED|ERROR' $L/${LANE}_$tag.log | tail -1)" >> $L/$LANE.log
done
echo DONE >> $L/$LANE.log
