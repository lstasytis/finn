#!/bin/bash
# Issue 024 A: block up to $1 s for the yolov8 feature test to finish; print progress.
L=/home/lstasytis/finn/build/finn_build/fx/fix024a/logs/yolo4_small_ana.log
D=/home/lstasytis/finn/build/finn_build/fx/fix024a/int2/rtlsim_finn_design_wrapper_sfks1p_k
timeout ${1:-580} bash -c "until grep -q ' passed in\| failed in' $L; do sleep 30; done"
echo "$(date +%H:%M) $(tail -1 $D/rtlsim_xsi_log.txt | cut -c1-60)"; grep -h " passed in\| failed in" $L; exit 0
