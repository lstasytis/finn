#!/bin/bash
# Issue 015: size every saved TAV model (fx/fix013-int/tav, 013's gen_tavs.sh) with the base
# sizer (pr-fifo worktree = feature/dev/analytical-fifo-sizing) and fix015; print the totals.
# usage: compare_all.sh <out dir> [model.onnx ...]   (PID in <out dir>/cmp.pid)
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/015-fifo-skip-edge-deadlock/scripts
O=/home/lstasytis/finn/build/finn_build/fx/fix013-int/tav
R=$1; shift; mkdir -p $R; echo $$ > $R/cmp.pid
M=${@:-$O/*.onnx}
for wt in pr-fifo fix015; do
  ( source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/$wt >/dev/null 2>&1
    export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix015
    for m in $M; do k=$(basename $m .onnx)
      while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
      [ -s $R/$k.$wt.txt ] || python $S/size_offline.py $m -v > $R/$k.$wt.txt 2>&1; done )
done
for m in $M; do
  k=$(basename $m .onnx)
  echo "$k | $(tail -n1 $R/$k.pr-fifo.txt) | $(tail -n1 $R/$k.fix015.txt)"
done > $R/summary.txt
echo DONE >> $R/summary.txt
