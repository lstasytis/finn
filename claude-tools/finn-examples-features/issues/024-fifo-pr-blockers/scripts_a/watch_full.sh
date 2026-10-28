#!/bin/bash
# Issue 024 A: exit when a FIFO deeper than 300 reaches its capacity in the live dump, or output appears.
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/024-fifo-pr-blockers/scripts_a
D=/home/lstasytis/finn/build/finn_build/fx/fix024a
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/fix024a >/dev/null 2>&1
while true; do
  last=$(tail -1 $D/live_mon/rtlsim_xsi_log.txt)
  echo "$last" | grep -q "m_axis_0=[1-9]" && { echo "OUTPUT $last"; exit 0; }
  r=$(python $S/live_full.py $D/yolo_copy/step_hw_ipgen.onnx $D/live_mon/maxcounts_live.txt 2>/dev/null | awk '$1 ~ /StreamingFIFO/ && $6 > 300 && $4 ~ /Concat|ElementwiseAdd|ElementwiseMul/')
  [ -n "$r" ] && { echo "FULL at $last"; echo "$r"; exit 0; }
  sleep 180
done
