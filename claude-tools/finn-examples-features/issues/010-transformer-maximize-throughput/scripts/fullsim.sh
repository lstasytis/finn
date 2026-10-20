#!/bin/bash
# full-graph stitched-IP rtlsim of a build with selected FIFOs enlarged (issue 009 node_sim.py)
# usage: fullsim.sh <model.onnx> <tag> <depth> [fifo,names|NONE] [frames]
cd /home/lstasytis/finn/build/wt/fix010
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix010
export NUM_DEFAULT_WORKERS=1 SWEEP_PART=xczu7ev-ffvc1156-2-e SWEEP_MAXITERS=200000
export SWEEP_FIFOS=${4:-}
L=$FINN_BUILD_DIR/logs010b
cd $L && python /home/lstasytis/finn/claude-tools/finn-examples-features/issues/009-tiled-mvau-exp-cycles/scripts/node_sim.py \
  $1 DEEPFIFO:$3 ${5:-4} > $L/full_$2.log 2>&1
grep -h RESULT $L/full_$2.log | sed "s/^/$2 /" >> $L/results.txt
