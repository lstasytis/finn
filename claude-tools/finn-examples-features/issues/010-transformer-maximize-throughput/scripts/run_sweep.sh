#!/bin/bash
# per-node stitched-IP rtlsim sweep, 6 at a time
cd /home/lstasytis/finn/build/wt/fix010
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix010
export NUM_DEFAULT_WORKERS=1
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/010-transformer-maximize-throughput/scripts
B=/home/lstasytis/finn/build/finn_build/fx/folding2/fold-max_transformer_transformer_benchmark_ZCU104_j33hxjs7
L=$FINN_BUILD_DIR/logs
cut -d' ' -f1 $L/nodelist.txt | xargs -P 6 -I{} bash -c "cd $L && timeout 1800 python $S/nodesim.py $B {} {} 4 > $L/node_{}.log 2>&1; grep -h RESULT $L/node_{}.log >> $L/results.txt"
echo SWEEP_DONE >> $L/results.txt
