#!/bin/bash
# Issue 018 (after 015's compare_all.sh): size every saved TAV model (fx/fix013-int/tav, plus
# the mobilenet full/small TAVs of fx/fix018/tav) with the sizer of each named worktree
# (size_offline.py = DeriveFIFOSizes); print the totals side by side. CMPJ (2) models at a time, one thread each.
# usage: compare_all.sh <out dir> <worktree> [<worktree> ...]   (PID in <out dir>/cmp.pid)
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/018-mobilenet-fifo-oversizing/scripts
R=$1; shift; mkdir -p $R; echo $$ > $R/cmp.pid
M="/home/lstasytis/finn/build/finn_build/fx/fix013-int/tav/*.onnx /home/lstasytis/finn/build/finn_build/fx/fix018/tav/mnv1_*.onnx"
for wt in "$@"; do
  ( source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/$wt >/dev/null 2>&1
    export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix018 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    for m in $M; do k=$(basename $m .onnx)
      while [ $(jobs -rp | wc -l) -ge ${CMPJ:-2} ]; do sleep 5; done
      while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
      [ -s $R/$k.$wt.txt ] || python $S/size_offline.py $m -v > $R/$k.$wt.txt 2>&1 &
    done; wait )
done
for m in $M; do
  k=$(basename $m .onnx); line="$k"
  for wt in "$@"; do line="$line | $(tail -n1 $R/$k.$wt.txt | awk '{print $5}')"; done
  echo "$line"
done > $R/summary.txt
echo DONE >> $R/summary.txt
