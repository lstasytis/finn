#!/bin/bash
# transformer_benchmark feature tests on fix013-int (fix/010 + fix/013), one at a time.
# usage: run_tb_tests.sh  (PID in $FINN_BUILD_DIR/logs/tb.pid)
cd /home/lstasytis/finn/build/wt/fix013-int
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export NUM_DEFAULT_WORKERS=4
L=$FINN_BUILD_DIR/logs
echo $$ > $L/tb.pid
T=tests/benchmark/transformer/test_build_transformer.py
for k in "folding and maximize" "fifo_sizing and heuristic_analytical" "folding and target"; do
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  tag=$(echo $k | tr ' ' '_')
  echo "$(date +%T) start $k" >> $L/tb.log
  timeout 7200 python -m pytest -p no:cacheprovider -v $T -k "transformer_benchmark and $k" \
    --junitxml=$L/tb_$tag.xml > $L/tb_$tag.log 2>&1
  echo "$(date +%T) rc=$? $k $(grep -E 'PASSED|FAILED' $L/tb_$tag.log | tail -1)" >> $L/tb.log
done
echo DONE >> $L/tb.log
