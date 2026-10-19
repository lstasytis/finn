#!/bin/bash
# throughput.sh <model_dir> <model|-> <variant>: config from the ablated sizer, then a full run
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
S=$(dirname $(dirname "$0")); T=$S/tools
md=$1; m=$2; v=$3; key=$m; [ "$m" = "-" ] && key=$md
F=$FINN_BUILD_DIR/fixtures/$key; O=$FINN_BUILD_DIR/variants/$key/$v; mkdir -p $(dirname $O)
python $T/ablated_config.py $md $m ZCU104 $F $S/ablations4/$v.py $O > $O.cfg.log 2>&1 || { echo "config FAILED: $O.cfg.log"; exit 1; }
python $T/run_variant.py fifo_abl_$v $md $m ZCU104 $O/final_hw_config.json > $O.run.log 2>&1
grep RESULT $O.run.log || { echo "run FAILED: $O.run.log"; exit 1; }
