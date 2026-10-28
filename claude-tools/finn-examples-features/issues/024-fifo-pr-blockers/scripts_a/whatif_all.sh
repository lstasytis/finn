#!/bin/bash
# Issue 024 A: whatif.py <variant> over every saved TAV model, J at a time. usage: whatif_all.sh <variant> <outdir> [J]
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/024-fifo-pr-blockers/scripts_a
V=$1; R=$2; J=${3:-6}; mkdir -p $R
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/fix024a >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix024a/tav OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in ${MODELS:-/home/lstasytis/finn/build/finn_build/fx/fix013-int/tav/*.onnx /home/lstasytis/finn/build/finn_build/fx/fix018/tav/mnv1_*.onnx /home/lstasytis/finn/build/finn_build/fx/fix024a/tav/*.onnx /home/lstasytis/finn/build/finn_build/fx/fix024a/tav_small/*.onnx}; do
  k=$(basename $m .onnx)
  while [ $(jobs -rp | wc -l) -ge $J ]; do sleep 3; done
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  [ -s $R/$k.txt ] || python $S/whatif.py $m $V 2>&1 | grep -v -i warn > $R/$k.txt &
done; wait; echo DONE > $R/DONE
