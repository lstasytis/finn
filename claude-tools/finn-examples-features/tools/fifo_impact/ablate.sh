#!/bin/bash
# Size every fixture with every ablation variant; print the edges that differ from baseline.
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
S=$(dirname $(dirname "$0")); T=$S/tools; A=$S/ablations; F=$FINN_BUILD_DIR/fixtures
python $T/make_ablations.py $A || exit 1
for v in $A/*.py; do
  name=$(basename $v .py)
  echo "##### $name"
  for d in $F/*/; do
    m=$(basename $d)
    [ -f $d/baseline.json ] || continue
    python $T/size_fixture.py $d $d/abl_$name.json $v > /dev/null 2>&1 || { echo "$m: sizing FAILED"; continue; }
    python $T/compare.py $d/baseline.json $d/abl_$name.json | sed "s|$F/||"
  done
done
