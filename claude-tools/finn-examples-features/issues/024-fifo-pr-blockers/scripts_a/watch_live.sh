#!/bin/bash
# Issue 024 A: exit when the live sim emits output, finishes, or its input % has not moved for 2M ticks.
L=$1
prev=""; prevt=0
while true; do
  last=$(tail -1 $L)
  echo "$last" | grep -q "m_axis_0=[1-9]" && { echo "OUTPUT $last"; exit 0; }
  grep -q TIMEOUT $L && { echo "DONE"; exit 0; }
  t=$(echo "$last" | sed 's/@\([0-9]*\) ticks.*/\1/'); s=$(echo "$last" | grep -o "s_axis_0=[0-9]*%")
  if [ "$s" != "$prev" ]; then prev=$s; prevt=$t; elif [ $((t - prevt)) -gt 2000000 ] && [ "$s" != "s_axis_0=100%" ]; then echo "STALL $last since $prevt"; exit 0; fi
  sleep 120
done
