#!/bin/bash
# Issue 018: run perf.py jobs from a job file, at most J at a time, each waiting for
# MemAvailable >= 200 GB. Job file lines: <tag> <copy dir> <perf.py args...>
# Results: $R/<tag>.json, logs $R/<tag>.log; PIDs in $R/pids.txt.
# usage: run_perf.sh <jobfile> <result dir> [J=8]
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/018-mobilenet-fifo-oversizing/scripts
JF=$1; R=$2; J=${3:-8}; mkdir -p $R
echo "runner $$" >> $R/pids.txt
cd /home/lstasytis/finn/build/wt/fix018
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix018/sim
export NUM_DEFAULT_WORKERS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while read -r tag cdir args; do
  [ -z "$tag" ] && continue; [[ $tag == \#* ]] && continue
  [ -s $R/$tag.json ] && continue
  while [ $(jobs -rp | wc -l) -ge $J ]; do sleep 20; done
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  python $S/perf.py $cdir OUT=$R/$tag.json $args < /dev/null > $R/$tag.log 2>&1 &
  echo "$tag $!" >> $R/pids.txt
  sleep 60
done < $JF
wait
echo DONE >> $R/pids.txt
