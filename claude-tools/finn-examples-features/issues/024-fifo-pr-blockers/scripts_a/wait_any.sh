#!/bin/bash
# Issue 024 A: wait (up to $1 s) for any pending result: live watcher, targeted sim, feature tests.
T=/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/tasks
D=/home/lstasytis/finn/build/finn_build/fx/fix024a
end=$(( $(date +%s) + ${1:-590} ))
seen0=$(cat $D/logs/{yolo4_small_ana,r18_small_ana,r18_full_ana,r50_small_ana}.log 2>/dev/null | grep -c " passed\| failed")
while [ $(date +%s) -lt $end ]; do
  [ -s $T/b8mqe8clt.output ] && { echo "LIVE: $(cat $T/b8mqe8clt.output)"; break; }
  [ -s $D/res/y_f151.json ] && { echo "F151 done"; break; }
  n=$(cat $D/logs/{yolo4_small_ana,r18_small_ana,r18_full_ana,r50_small_ana}.log 2>/dev/null | grep -c " passed\| failed")
  [ "$n" != "$seen0" ] && { echo "TEST finished"; break; }
  sleep 30
done
date; echo "live: $(tail -1 $D/live_mon/rtlsim_xsi_log.txt | cut -c1-60)"
grep -h " passed\| failed" $D/logs/{yolo4_small_ana,r18_small_ana,r18_full_ana,r50_small_ana}.log 2>/dev/null
