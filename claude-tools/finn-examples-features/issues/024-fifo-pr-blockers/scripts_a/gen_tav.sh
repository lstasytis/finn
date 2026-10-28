#!/bin/bash
# Issue 024 A: TAV model of a build (013's tav_model.py). usage: gen_tav.sh <build dir> <out.onnx>
cd /home/lstasytis/finn/build/wt/fix024a
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix024a/tav
export NUM_DEFAULT_WORKERS=8
cd $FINN_BUILD_DIR
exec python /home/lstasytis/finn/claude-tools/finn-examples-features/issues/024-fifo-pr-blockers/scripts_a/tav_model.py "$@"
