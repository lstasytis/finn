#!/bin/bash
# usage: run.sh <model.onnx> <tag> <depth> <Ns>
cd /home/lstasytis/finn/build/wt/devsync-perf
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/devsync_perf
mkdir -p $FINN_BUILD_DIR/logs
python /home/lstasytis/finn/claude-tools/finn-examples-features/issues/025-devsync-throughput/scripts/chain_perf.py "$@" > $FINN_BUILD_DIR/logs/$2_chain.log 2>&1
grep RESULT $FINN_BUILD_DIR/logs/$2_chain.log || tail -30 $FINN_BUILD_DIR/logs/$2_chain.log
