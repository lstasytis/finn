#!/bin/bash
# gate.sh <snapshot dir> <out dir> <cpus> [driver args]
SNAP=$1; OUT=$2; CPUS=$3; shift 3
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/fix005 >/dev/null 2>&1
export PYTHONPATH=$SNAP/src:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1
exec python /home/lstasytis/finn/claude-tools/finn-examples-features/agents/folding_refactor/dump_folding.py \
  --impl $SNAP/src/finn/transformation/fpgadataflow/set_folding.py --out "$OUT" --cpus "$CPUS" "$@"
