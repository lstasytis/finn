#!/bin/bash
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
S=$(dirname $(dirname "$0")); T=$S/tools; F=$FINN_BUILD_DIR/fixtures
echo "=== regress on all fixtures"; $T/regress.sh
rm -rf $S/ablations2; python $T/make_ablations.py $S/ablations2
one() { v=$1; python $T/size_fixture.py $F/resnet50 $F/resnet50/abl2_$v.json $S/ablations2/$v.py > $F/resnet50/abl2_$v.log 2>&1 || echo "##### $v FAILED"; }
export -f one; export T F S
printf "%s\n" no_transparent_ewadd no_idle_floor no_quirk_inherit no_join_rule no_fill_only no_slack no_burst_floor no_cap no_pacer_tol no_pacer_reset no_input_pacing no_frame_split | xargs -P 4 -I{} bash -c 'one {}'
for v in no_transparent_ewadd no_idle_floor no_quirk_inherit no_join_rule no_fill_only no_slack no_burst_floor no_cap no_pacer_tol no_pacer_reset no_input_pacing no_frame_split; do
  [ -f $F/resnet50/abl2_$v.json ] || continue
  echo "##### $v"; python $T/compare.py $F/resnet50/baseline.json $F/resnet50/abl2_$v.json | tail -12
done
