#!/bin/bash
# Timed builds from a job list, one worker per CPU lane (whole CCDs with their SMT siblings, own
# Vivado slot directory and slot count), so that builds side by side share no cores, caches or
# Vivado slots. A memory watchdog runs next to the workers (mem_watchdog.sh).
#
#   JOBS=<file> [LANES="0-23,64-87 24-47,88-111"] [LANE_SLOTS=32] setsid nohup run_campaign.sh &
#
# Job line: board clk dir name mode
#   board  ZCU104 | U55C;  clk  ns;  mode  islands | global | vivado
#   dir    the model directory: <dir>/<name>/frontend/intermediate_models/step_set_fifo_depths.onnx
#          or the path in <dir>/<name>.model; results in <dir>/vshell_timing/runs.jsonl
# Jobs are taken in order (lines appended later are picked up too). An island or global build
# that had to build its shell first (shell status "built") is repeated.
set -u
FINN_BUILD_DIR=${RWSCALE_BUILD_DIR:-/home/lstasytis/finn/build/finn_build}
: "${PLATFORM_REPO_PATHS:=/mnt/labstore/Xilinx/2025.1/Vitis/platforms}"
export FINN_BUILD_DIR PLATFORM_REPO_PATHS
HERE=$(cd "$(dirname "$0")" && pwd)
JOBS=${JOBS:?job list}
LANES=${LANES:-"0-23,64-87 24-47,88-111"}
LANE_SLOTS=${LANE_SLOTS:-32}
STATE=${JOBS%.txt}
LOG=$STATE.log
[ -f $STATE.next ] || echo 0 > $STATE.next

next_job() {
  (
    flock 9
    local k
    k=$(cat $STATE.next)
    echo $((k + 1)) > $STATE.next
    sed -n "$((k + 1))p" $JOBS
  ) 9> $STATE.lock
}

lane_worker() {  # lane index, CPU list
  local i=$1 cpus=$2 job board clk dir n mode last threads
  threads=$(python3 -c "import sys; print(sum(int(b) - int(a) + 1 for a, b in (r.split('-') for r in sys.argv[1].split(','))))" $cpus)
  while true; do
    job=$(next_job)
    [ -z "$job" ] && break
    read board clk dir n mode <<< "$job"
    if [ ! -f $dir/$n/frontend/intermediate_models/step_set_fifo_depths.onnx ] && [ ! -f $dir/$n.model ]; then
      echo "$(date +%T) lane $i: skip $job (no model)" >> $LOG
      continue
    fi
    case $board in
      ZCU104|ZCU102|Pynq-Z1|KV260_SOM) shells=$FINN_BUILD_DIR/rwislands/shells ;;
      *) shells=$FINN_BUILD_DIR/rwislands/vshells ;;
    esac
    for attempt in 1 2; do
      echo "$(date +%T) lane $i ($cpus): start $job" >> $LOG
      D=$dir CLK=$clk BOARD=$board SHELLS=$shells MODELS=$n MODE=$mode WORKERS=$threads \
        DYNARAPID_VIVADO_SLOTS=$LANE_SLOTS DYNARAPID_VIVADO_SLOTS_DIR=$STATE.slots_lane$i \
        taskset -c $cpus bash $HERE/run_vshell_timing.sh > $dir/lane${i}_${n}_${mode}.out 2>&1
      last=$(grep "\"model\": \"$n\", \"mode\": \"$mode\"" $dir/vshell_timing/runs.jsonl | tail -n 1)
      echo "$(date +%T) lane $i: done $job: $(echo "$last" | cut -c1-200)" >> $LOG
      echo "$last" | grep -q '"shell": "built"' || break
      echo "$(date +%T) lane $i: $job built its shell, repeating" >> $LOG
    done
  done
}

echo "$(date +%T) campaign $JOBS on lanes $LANES" >> $LOG
bash $HERE/mem_watchdog.sh $$ $STATE.watchdog.log 20 &
i=0
pids=""
for cpus in $LANES; do
  lane_worker $i $cpus &
  pids="$pids $!"
  i=$((i + 1))
done
# (the workers only: the watchdog lives as long as this script)
wait $pids
echo "$(date +%T) campaign done" >> $LOG
