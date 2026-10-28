#!/bin/bash
# Issue 024 A: one feature test in build/wt/fix024a-int. usage: run_test.sh <tag> <test file> <-k expr> [-m marker]
tag=$1; tf=$2; k=$3; mk=${4:-fifo_sizing}
L=/home/lstasytis/finn/build/finn_build/fx/fix024a/logs; mkdir -p $L
cd /home/lstasytis/finn/build/wt/${WT:-fix024a-int}
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix024a/${BD:-int} NUM_DEFAULT_WORKERS=4
mkdir -p $FINN_BUILD_DIR
echo "$tag $$" >> $L/pids.txt
exec python -m pytest -p no:cacheprovider -v $tf -m $mk -k "$k" --junitxml=$L/$tag.xml > $L/$tag.log 2>&1
