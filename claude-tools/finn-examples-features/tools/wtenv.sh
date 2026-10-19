# Environment for running FINN from a worktree: source wtenv.sh <worktree dir>
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/env24.sh
export FINN_ROOT=$(realpath "${1:-$PWD}")
export PYTHONPATH=$FINN_ROOT/src:$FINN_ROOT/tests:$PYTHONPATH
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/$(basename "$FINN_ROOT")
mkdir -p "$FINN_BUILD_DIR"
