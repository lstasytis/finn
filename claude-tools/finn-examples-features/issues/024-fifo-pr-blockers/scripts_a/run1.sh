#!/bin/bash
# Issue 024 A: run one perf.py job in the fix024a worktree env. usage: run1.sh <tag> <copy dir> <perf args...>
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/024-fifo-pr-blockers/scripts_a
R=/home/lstasytis/finn/build/finn_build/fx/fix024a/res; mkdir -p $R
tag=$1; shift; cdir=$1; shift
cd /home/lstasytis/finn/build/wt/fix024a
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix024a/sim
export NUM_DEFAULT_WORKERS=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$tag $$" >> $R/pids.txt
exec python $S/perf.py $cdir OUT=$R/$tag.json "$@" < /dev/null > $R/$tag.log 2>&1
