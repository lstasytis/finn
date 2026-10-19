#!/bin/bash
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
S=$(dirname $(dirname "$0")); T=$S/tools; F=$FINN_BUILD_DIR/fixtures
rm -rf $S/ablations3; python $T/make_ablations.py $S/ablations3
cp $FINN_ROOT/src/finn/transformation/fpgadataflow/derive_characteristic.py $S/ablations3/current.py
one() { v=$1; python $T/size_fixture.py $F/resnet50 $F/resnet50/abl3_$v.json $S/ablations3/$v.py > $F/resnet50/abl3_$v.log 2>&1 || echo "##### $v FAILED"; }
export -f one; export T F S
printf "%s\n" current no_transparent_ewadd no_quirk_inherit no_idle_floor no_join_rule no_fill_only | xargs -P 6 -I{} bash -c 'one {}'
for v in current no_transparent_ewadd no_quirk_inherit no_idle_floor no_join_rule no_fill_only; do
  [ -f $F/resnet50/abl3_$v.json ] || continue
  echo "##### $v"; python $T/compare.py $F/resnet50/baseline.json $F/resnet50/abl3_$v.json | tail -14
done
