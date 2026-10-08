#!/bin/bash
# Build time vs accelerator size: VGG10 and MobileNet on the U55C (Vivado-only shell), FINN's
# global Vivado flow vs the island flow, while the folding scales the parallelism; at 100, 200
# and 300 MHz.
#
#   setsid nohup experiments/dynarapid/run_parallelism_scaling.sh &   (all clocks, all points)
#   CLOCKS=10 POINTS="vgg10:1:r" experiments/dynarapid/run_parallelism_scaling.sh
#   python experiments/dynarapid/summarize_parallelism_scaling.py     (tables + plots)
#
# Points: model:factor:mode. factor x the throughput of the model's hand-written folding
# (VGG10 finn-examples, MobileNet U250) is the target of FINN's automatic folding (SetFolding,
# scaling_folding.py), scaled with the clock so that a point has the same folding at every
# clock (same cycles per frame). mode r = balanced (two-pass relaxation: every layer folded for
# the bottleneck's throughput), n = no relaxation. FINN's sliding windows produce at most one
# output pixel per cycle, so both models saturate near factor 1; beyond that only n grows the
# design (the other layers keep unfolding towards the unreachable target): more resources, same
# throughput. The plots use the synthesized size as x-axis.
#
# Phase 1 (untimed, nothing timed runs meanwhile): all frontends (up to FIFO sizing) and the
# shells of every clock. Phase 2 (timed): one job = one build (model point, flow, clock), run
# under taskset on one of the CPU lanes, each lane a set of whole CCDs (own L3) with its SMT
# siblings, its own Vivado slot directory and slot count (memory split between the lanes), so
# that builds running side by side do not share cores, caches or slots. Every build of the
# experiment runs on a lane of the same size (results comparable across points and clocks).
# Jobs: 100 MHz first, then 200, then 300; small points first, models interleaved. An island run
# that had to build its shell (status "built") is repeated.
set -u
# (the container's default FINN_BUILD_DIR, /tmp/finn_dev_lstasytis, is too small and has no
# cached shells: this experiment always uses the repo volume unless RWSCALE_BUILD_DIR says otherwise)
FINN_BUILD_DIR=${RWSCALE_BUILD_DIR:-/home/lstasytis/finn/build/finn_build}
: "${PLATFORM_REPO_PATHS:=/mnt/labstore/Xilinx/2025.1/Vitis/platforms}"
export FINN_BUILD_DIR PLATFORM_REPO_PATHS
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
D=${D:-$FINN_BUILD_DIR/rwscale}
CLOCKS=${CLOCKS:-"10 5 3.333"}
POINTS=${POINTS:-"vgg10:0.25:r mnv1:0.25:r vgg10:0.5:r mnv1:0.5:r vgg10:1:r mnv1:1:r vgg10:1.25:n mnv1:1.25:n vgg10:3:n mnv1:2:n vgg10:8:n"}
MODES=${MODES:-"islands vivado"}
# EPYC 9554P: CCD k = CPUs 8k..8k+7 and their SMT siblings 64+8k..64+8k+7
LANES=${LANES:-"0-31,64-95 32-63,96-127"}
LANE_THREADS=${LANE_THREADS:-64}
# memory: ~4.5 GB per Vivado run, 755 GB split between the lanes (the island flow's default
# would size its slots from the whole machine's free memory)
LANE_SLOTS=${LANE_SLOTS:-40}
FRONTEND_JOBS=${FRONTEND_JOBS:-8}
VGG10_FPS=97276.26   # hand-written folding at 100 MHz (estimate report)
MNV1_FPS=898.15
SHELLS=$FINN_BUILD_DIR/rwislands/vshells
# accelerators of earlier island runs: their IODMA / AXI-Lite interfaces key the shells
SHELL_REFS="vgg10:$FINN_BUILD_DIR/rwu/vshell_timing/vgg10_islands/islands/accel.onnx mnv1:$FINN_BUILD_DIR/rwu/vshell_timing/mnv1_islands/islands/accel.onnx"
mkdir -p $D
LOG=$D/scaling.log

name_of() { local m=${1%%:*} rest=${1#*:}; echo "${m}_s${rest%%:*}${rest##*:}"; }

frontend() {  # clk point
  local clk=$1 p=$2 m=${2%%:*} rest=${2#*:} n DC
  local f=${rest%%:*} mode=${rest##*:}
  n=$(name_of $p); DC=$D/clk$clk
  [ -f $DC/$n/frontend/intermediate_models/step_set_fifo_depths.onnx ] && return 0
  mkdir -p $DC
  local relax=""; [ "$mode" = n ] && relax="--no-relax"
  # (HLS / codegen parallelism of one frontend; several frontends run side by side)
  export NUM_DEFAULT_WORKERS=${FRONTEND_WORKERS:-16}
  if [ $m = vgg10 ]; then
    python $HERE/run_vgg10.py --model $ROOT/tests/benchmark/models/radioml_w4a4_small_tidy.onnx \
      --out $DC/$n --mode frontend --board U55C --clk $clk \
      --target-fps $(python3 -c "print($VGG10_FPS * $f * 10 / $clk)") $relax > $DC/${n}_frontend.log 2>&1
  else
    python $HERE/run_mobilenet.py --out $DC/$n --mode frontend --board U55C --clk $clk \
      --target-fps $(python3 -c "print($MNV1_FPS * $f * 10 / $clk)") $relax > $DC/${n}_frontend.log 2>&1
  fi
  echo "$(date +%T) frontend clk $clk $n rc=$?" >> $LOG
}

if [ -z "${SKIP_PHASE1:-}" ]; then
  echo "$(date +%T) phase 1: shells and frontends (clocks $CLOCKS)" >> $LOG
  for clk in $CLOCKS; do
    for r in $SHELL_REFS; do
      python $HERE/build_vshell.py ${r#*:} $clk $SHELLS > $D/shell_${r%%:*}_clk$clk.log 2>&1 &
    done
  done
  for clk in $CLOCKS; do
    for p in $POINTS; do
      while [ $(jobs -rp | wc -l) -ge $((FRONTEND_JOBS + 6)) ]; do sleep 5; done
      frontend $clk $p &
    done
  done
  wait
  echo "$(date +%T) phase 1 done: $(grep -h '"status"' $D/shell_*.log | tr '\n' ' ')" >> $LOG
fi

# phase 2: the job list, consumed by one worker per lane
JOBS=$D/jobs.txt
: > $JOBS
for clk in $CLOCKS; do
  for p in $POINTS; do
    for mode in $MODES; do echo "$clk $(name_of $p) $mode" >> $JOBS; done
  done
done
echo 0 > $D/jobs.next

next_job() {
  (
    flock 9
    local k
    k=$(cat $D/jobs.next)
    echo $((k + 1)) > $D/jobs.next
    sed -n "$((k + 1))p" $JOBS
  ) 9> $D/jobs.lock
}

lane_worker() {  # lane index, CPU list
  local i=$1 cpus=$2 job clk n mode DC last
  while true; do
    job=$(next_job)
    [ -z "$job" ] && break
    read clk n mode <<< "$job"
    DC=$D/clk$clk
    if [ ! -f $DC/$n/frontend/intermediate_models/step_set_fifo_depths.onnx ]; then
      echo "$(date +%T) lane $i: skip $job (no frontend)" >> $LOG
      continue
    fi
    for attempt in 1 2; do
      echo "$(date +%T) lane $i ($cpus): start $job" >> $LOG
      D=$DC CLK=$clk MODELS=$n MODE=$mode WORKERS=$LANE_THREADS \
        DYNARAPID_VIVADO_SLOTS=$LANE_SLOTS DYNARAPID_VIVADO_SLOTS_DIR=$D/slots_lane$i \
        taskset -c $cpus bash $HERE/run_vshell_timing.sh > $DC/lane${i}_${n}_${mode}.out 2>&1
      last=$(grep "\"model\": \"$n\", \"mode\": \"$mode\"" $DC/vshell_timing/runs.jsonl | tail -n 1)
      echo "$(date +%T) lane $i: done $job: $(echo "$last" | cut -c1-160)" >> $LOG
      echo "$last" | grep -q '"shell": "built"' || break
      echo "$(date +%T) lane $i: $job built its shell, repeating" >> $LOG
    done
  done
}

echo "$(date +%T) phase 2: $(wc -l < $JOBS) timed builds on lanes $LANES" >> $LOG
i=0
for cpus in $LANES; do
  lane_worker $i $cpus &
  i=$((i + 1))
done
wait
echo "$(date +%T) done" >> $LOG
