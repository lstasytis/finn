#!/bin/bash
# Size every fixture with the worktree's derive_characteristic (or $1 = module path)
# and compare against baseline.json. Usage: regress.sh [module.py] [tag]
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
T=$(dirname "$0"); F=$FINN_BUILD_DIR/fixtures; TAG=${2:-current}
rc=0
for d in $F/*/; do
  m=$(basename $d)
  [ -f $d/baseline.json ] || continue
  python $T/size_fixture.py $d $d/$TAG.json $1 > /dev/null 2>&1 || { echo "$m: sizing FAILED"; rc=1; continue; }
  python $T/compare.py $d/baseline.json $d/$TAG.json || rc=1
done
exit $rc
