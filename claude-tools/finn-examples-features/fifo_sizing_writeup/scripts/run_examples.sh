#!/bin/bash
# Reproduce the numbers of fifo_sizing_analytical.tex.
#   run_examples.sh <worktree of feature/dev/analytical-fifo-sizing> [example ...] [--all-passes]
# e.g. (worktree at commit 39dac9a18):
#   git -C /home/lstasytis/finn worktree add build/wt/fifodoc --detach feature/dev/analytical-fifo-sizing
#   claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fifodoc
#   run_examples.sh /home/lstasytis/finn/build/wt/fifodoc --all-passes > fifo_examples_output.txt
# Pure Python (tree models and hand-made TAVs): no synthesis, no rtlsim.
set -e
HERE=$(dirname "$(realpath "$0")")
WT=$(realpath "$1"); shift
cd "$WT"
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh "$WT" >/dev/null 2>&1
export FINN_BUILD_DIR=${FINN_BUILD_DIR:-/home/lstasytis/finn/build/finn_build/fx/$(basename "$WT")}
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=$WT/tests/fpgadataflow:$PYTHONPATH
python -W ignore "$HERE/fifo_examples.py" "$@"
