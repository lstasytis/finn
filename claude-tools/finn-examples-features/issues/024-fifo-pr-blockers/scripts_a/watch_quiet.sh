#!/bin/bash
# Issue 024 A: exit when maxcount_262 (Concat_12 in0) > 0, output appears, or no maxcount changed for 3M cycles.
D=/home/lstasytis/finn/build/finn_build/fx/fix024a/live_mon
prev=""; since=""
while true; do
  cur=$(tail -n +2 $D/maxcounts_live.txt | md5sum); at=$(head -1 $D/maxcounts_live.txt | tr -d @)
  [ -z "$at" ] && { sleep 30; continue; }
  grep -q "m_axis_0=[1-9]" <(tail -1 $D/rtlsim_xsi_log.txt) && { echo "OUTPUT at $at"; exit 0; }
  v=$(awk '$1=="maxcount_262" {print $2}' $D/maxcounts_live.txt); [ -n "$v" ] && [ "$v" -gt 0 ] && { echo "IN0 arrived at $at ($v)"; exit 0; }
  if [ "$cur" != "$prev" ]; then prev=$cur; since=$at; elif [ $((at - since)) -ge 3000000 ]; then echo "QUIET since $since (now $at)"; cp $D/maxcounts_live.txt $D/snap_quiet.txt; exit 0; fi
  sleep 60
done
