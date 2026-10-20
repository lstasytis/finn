#!/bin/bash
# run_eval.sh <worktree> <tag> <pads e.g. 0,4> <spec>...: eval_build.py per spec "testdir:board:model"
# (model "-" for single-model files), all specs x pads in parallel (max 14 at once),
# JSON+log in $FINN_BUILD_DIR/eval/<tag>_<model>_p<pad>.*
WT=$1; TAG=$2; PADS=$3; shift 3
cd $WT && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export OMP_NUM_THREADS=1 NUM_DEFAULT_WORKERS=1
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/014-dwc-padding-tightness/scripts
mkdir -p $FINN_BUILD_DIR/eval
for spec in "$@"; do
  IFS=: read tdir board model <<< "$spec"
  name=$([ "$model" = "-" ] && basename $tdir || echo $model)_$board
  for pad in ${PADS//,/ }; do
    while [ $(jobs -rp | wc -l) -ge 14 ]; do sleep 5; done
    while [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt 220 ]; do sleep 30; done
    timeout 14400 python $S/eval_build.py $tdir $board $model $pad ${SEED:-1} \
      $FINN_BUILD_DIR/eval/${TAG}_${name}_p$pad.json > $FINN_BUILD_DIR/eval/${TAG}_${name}_p$pad.log 2>&1 &
    echo "$name p$pad pid $!"
  done
done
wait
