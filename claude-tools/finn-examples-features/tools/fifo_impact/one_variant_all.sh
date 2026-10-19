#!/bin/bash
# one_variant_all.sh <variant>: regenerate ablations4 and size every fixture with one variant
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
S=$(dirname $(dirname "$0")); T=$S/tools; F=$FINN_BUILD_DIR/fixtures; v=$1
python $T/make_ablations.py $S/ablations4 > /dev/null
for d in $F/*/; do m=$(basename $d); [ -f $d/baseline.json ] || continue
  python $T/size_fixture.py $d $d/abl4_$v.json $S/ablations4/$v.py > /dev/null 2>&1 || { echo "$m: FAILED"; continue; }
  python $T/compare.py $d/baseline.json $d/abl4_$v.json | tail -1 | sed "s|$F/||"
done
