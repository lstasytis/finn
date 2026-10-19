#!/bin/bash
source /tmp/finn_dev_lstasytis/agents/folding_fix/env.sh
cd /tmp/finn_dev_lstasytis/worktrees/folding-fix
L=/tmp/finn_dev_lstasytis/agents/folding_fix/logs
R=$FINN_BUILD_DIR/feature_flow_results
: > $R/padding_tightness.jsonl
pytest tests/benchmark/test_dwc_padding_flow.py -k "tightness and ZCU104" -n 4 -p no:cacheprovider > $L/fix3_zcu104.log 2>&1
cp $R/padding_tightness.jsonl $L/tightness_fix3_ZCU104.jsonl
: > $R/padding_tightness.jsonl
pytest tests/benchmark/test_dwc_padding_flow.py -k "tightness and U55C" -n 4 -p no:cacheprovider > $L/fix3_u55c.log 2>&1
cp $R/padding_tightness.jsonl $L/tightness_fix3_U55C.jsonl
echo "TIGHTNESS DONE" >> $L/chain.status
FEATURE_FLOW_QUICK=1 pytest tests/benchmark/test_folding_flow.py tests/benchmark/test_dwc_padding_flow.py -k "padding_flow or folding_flow" -n 4 -p no:cacheprovider > $L/quick_flows.log 2>&1
echo "QUICK DONE" >> $L/chain.status
pytest tests/benchmark/test_dwc_padding_flow.py -k "padding_flow and (tfc-w2a2 or cnv-w2a2) and ZCU104" -n 2 -p no:cacheprovider > $L/full_flows.log 2>&1
echo "FULL DONE" >> $L/chain.status
