#!/bin/bash
# Issue 024 A: block up to $1 s for: a feature test finishing, y_f151 result, live sim output.
D=/home/lstasytis/finn/build/finn_build/fx/fix024a
L="$D/logs/yolo4_small_ana.log $D/logs/r18_small_ana.log $D/logs/r18_full_ana.log $D/logs/r50_small_ana.log"
end=$(( $(date +%s) + ${1:-580} ))
n0=$(cat $L 2>/dev/null | grep -c " passed in\| failed in\|error in")
while [ $(date +%s) -lt $end ]; do
  n=$(cat $L 2>/dev/null | grep -c " passed in\| failed in\|error in")
  [ "$n" != "$n0" ] && { echo "TEST finished"; break; }
  [ -s $D/res/y_f151.json ] && { echo "F151 done"; break; }
  tail -1 $D/live_mon/rtlsim_xsi_log.txt | grep -q "m_axis_0=[1-9]" && { echo "LIVE output"; break; }
  sleep 30
done
echo "$(date +%H:%M) live $(tail -1 $D/live_mon/rtlsim_xsi_log.txt | cut -c1-30) f151 $(tail -1 $D/sim/rtlsim_finn_design_wrapper_f92f4j_6/rtlsim_xsi_log.txt | cut -c1-45) cnt $(tail -1 $D/live_cnt/rtlsim_xsi_log.txt | cut -c1-30)"
grep -h " passed in\| failed in" $L 2>/dev/null
exit 0
