#!/bin/bash
source /tmp/finn_dev_lstasytis/agents/folding_fifo/env.sh
cd /tmp/finn_dev_lstasytis/worktrees/folding-fifo
L=/tmp/finn_dev_lstasytis/agents/folding_fifo/logs; S=/tmp/claude-1205/-home-lstasytis-finn-dev-lstasytis-finn/8871d419-69de-49fc-824c-30ba6c67e801/scratchpad
R=$FINN_BUILD_DIR/feature_flow_results
: > $R/padding_tightness.jsonl
pytest tests/benchmark/test_dwc_padding_flow.py -k "tightness and U55C" -n 4 -p no:cacheprovider > $L/tightness_u55c.log 2>&1
cp $R/padding_tightness.jsonl $L/tightness_final_U55C.jsonl
echo "U55C DONE" >> $L/chain.status
FEATURE_FLOW_QUICK=1 pytest tests/benchmark/test_folding_flow.py -n 4 -p no:cacheprovider > $L/quick_folding_flows.log 2>&1
echo "QUICK DONE" >> $L/chain.status
python $S/fold_fifo_cmp_fifo.py final optimizer,optimizer+fifo cnv-w1a1,cnv-w2a2,tfc-w2a2 > $L/final_cmp.log 2>&1
echo "CMP DONE" >> $L/chain.status
df -h /tmp/finn_dev_lstasytis >> $L/chain.status
python $S/full_cnv.py > $L/full_cnv.log 2>&1
echo "FULL DONE" >> $L/chain.status
