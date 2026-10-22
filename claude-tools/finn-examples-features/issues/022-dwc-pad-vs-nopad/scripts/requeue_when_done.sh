#!/bin/bash
# Requeue campaign jobs (requeue.sh) once their running pytest process has exited:
#   requeue_when_done.sh <job id>=<pid> ...
# Jobs are handled one at a time, 60 s after their process ended (the runner reaps it first).
T=/home/lstasytis/finn/claude-tools/finn-examples-features/tools
declare -A PIDS
for a in "$@"; do PIDS[${a%%=*}]=${a#*=}; done
while [ ${#PIDS[@]} -gt 0 ]; do
  for j in "${!PIDS[@]}"; do
    if ! kill -0 ${PIDS[$j]} 2>/dev/null; then
      sleep 60
      echo "$(date '+%F %T') requeue $j (pid ${PIDS[$j]} ended)"
      $T/requeue.sh $j
      unset "PIDS[$j]"
    fi
  done
  sleep 60
done
echo "$(date '+%F %T') all requeued"
