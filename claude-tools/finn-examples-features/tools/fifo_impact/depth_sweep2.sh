#!/bin/bash
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
S=$(dirname $(dirname "$0")); T=$S/tools; F=$FINN_BUILD_DIR/fixtures
echo "=== regress"; $T/regress.sh
rm -rf $S/ablations4; python $T/make_ablations.py $S/ablations4
# 9 linear models: every variant, sequential (fast)
for v in $S/ablations4/*.py; do
  name=$(basename $v .py); echo "##### $name"
  for d in $F/*/; do m=$(basename $d); [ "$m" = resnet50 ] && continue; [ -f $d/baseline.json ] || continue
    python $T/size_fixture.py $d $d/abl4_$name.json $v > /dev/null 2>&1 || { echo "$m: FAILED"; continue; }
    python $T/compare.py $d/baseline.json $d/abl4_$name.json | tail -1 | sed "s|$F/||"
  done
done
# resnet50: 4 in parallel
one() { v=$1; python $T/size_fixture.py $F/resnet50 $F/resnet50/abl4_$v.json $S/ablations4/$v.py > $F/resnet50/abl4_$v.log 2>&1 || echo "##### resnet50 $v FAILED"; }
export -f one; export T F S
ls $S/ablations4/*.py | xargs -n1 basename | sed 's/.py$//' | xargs -P 4 -I{} bash -c 'one {}'
for v in $S/ablations4/*.py; do name=$(basename $v .py); echo "##### resnet50 $name"; python $T/compare.py $F/resnet50/baseline.json $F/resnet50/abl4_$name.json | tail -1; done
