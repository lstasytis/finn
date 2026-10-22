#!/bin/bash
# Issue 018: mobilenet_v1 U55C heuristic_analytical FIFO feature tests (small, full) on
# fix/018-int, one pytest each, in parallel. Logs and PIDs in fx/fix018-int/logs.
W=/home/lstasytis/finn/build/wt/fix018-int
cd $W && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export NUM_DEFAULT_WORKERS=2 OMP_NUM_THREADS=1
L=$FINN_BUILD_DIR/logs; mkdir -p $L
for size in ${@:-small full}; do
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  python -m pytest -p no:cacheprovider -v tests/benchmark/mobilenet_v1/test_build_mobilenet_v1.py \
    -m fifo_sizing -k "heuristic_analytical and $size" --junitxml=$L/mnv1_$size.xml < /dev/null > $L/mnv1_$size.log 2>&1 &
  echo "$size $!" >> $L/pids.txt
done
wait
echo DONE >> $L/pids.txt
