#!/bin/bash
# Size every saved TAV model with the base sizer (fix010 worktree = feature/dev/analytical-fifo-sizing
# c74d17dc5) and the new one (fix013-int); print totals and the changed edges.
# usage: compare_all.sh <out dir> [model.onnx ...]
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/013-fifo-latency-model/scripts
O=/home/lstasytis/finn/build/finn_build/fx/fix013-int/tav
R=$1; shift; mkdir -p $R
M=${@:-$O/*.onnx}
for wt in fix010 fix013-int; do
  ( source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/$wt >/dev/null 2>&1
    export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix013-int
    for m in $M; do k=$(basename $m .onnx); python $S/size_offline.py $m -v > $R/${k}.$wt.txt 2>&1; done )
done
for m in $M; do
  k=$(basename $m .onnx)
  a=$(tail -n1 $R/$k.fix010.txt); b=$(tail -n1 $R/$k.fix013-int.txt)
  echo "$k | $a | $b"
done
