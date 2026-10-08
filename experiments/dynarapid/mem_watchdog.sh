#!/bin/bash
# Memory watchdog for long unattended experiments: if MemAvailable falls below LIMIT_GB, the
# largest Vivado / Vitis HLS / Java / xsim process is killed (that build fails and is logged)
# instead of letting the container run out of memory, which takes the whole machine down.
# Exits when the watched process (PID) is gone.
#
#   setsid nohup experiments/dynarapid/mem_watchdog.sh <pid> <log> [LIMIT_GB] &
set -u
WATCH=$1
LOG=$2
LIMIT_GB=${3:-20}
while kill -0 $WATCH 2>/dev/null; do
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  if [ $avail -lt $LIMIT_GB ]; then
    # largest resident tool process (never the MCP server or a shell)
    victim=$(ps -eo pid=,rss=,comm=,args= --sort=-rss \
      | awk '$3 ~ /^(vivado|vitis_hls|java|xsimk|xelab)$/ {print $1; exit}')
    if [ -n "$victim" ]; then
      echo "$(date '+%F %T') MemAvailable ${avail} GB < ${LIMIT_GB} GB: killing $(ps -o pid=,rss=,args= -p $victim | cut -c1-200)" >> $LOG
      kill -9 $victim
      sleep 10
    fi
  fi
  sleep 3
done
