#!/bin/bash
# Issue 024 A: size every TAV model with worktrees A and B (size_dump.py), J at a time; diff.
# usage: cmp_wt.sh <outdir> <wtA> <wtB> [J]
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/024-fifo-pr-blockers/scripts_a
R=$1; A=$2; B=$3; J=${4:-6}; mkdir -p $R
M="/home/lstasytis/finn/build/finn_build/fx/fix013-int/tav/*.onnx /home/lstasytis/finn/build/finn_build/fx/fix018/tav/mnv1_*.onnx /home/lstasytis/finn/build/finn_build/fx/fix024a/tav/*.onnx /home/lstasytis/finn/build/finn_build/fx/fix024a/tav_small/*.onnx"
for wt in $A $B; do
  ( source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/$wt >/dev/null 2>&1
    export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix024a/tav OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    for m in $M; do k=$(basename $(dirname $m))_$(basename $m .onnx)
      while [ $(jobs -rp | wc -l) -ge $J ]; do sleep 3; done
      while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
      [ -s $R/$k.$wt.json ] || python $S/size_dump.py $m $R/$k.$wt.json > $R/$k.$wt.log 2>&1 &
    done; wait )
done
for f in $R/*.$A.json; do k=$(basename $f .$A.json); python3 $S/size_diff.py $f $R/$k.$B.json $k; done > $R/summary.txt 2>&1
echo DONE >> $R/summary.txt
