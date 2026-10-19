#!/bin/bash
source /tmp/claude-1205/-home-lstasytis-finn-dev-lstasytis-finn/8871d419-69de-49fc-824c-30ba6c67e801/scratchpad/env.sh
cd /home/lstasytis/finn_dev_lstasytis/finn
L=/tmp/finn_dev_lstasytis/logs
mkdir -p /tmp/finn_dev_lstasytis/feature_flow_results_2023.1
mv /tmp/finn_dev_lstasytis/feature_flow_results/* /tmp/finn_dev_lstasytis/feature_flow_results_2023.1/ 2>/dev/null
T="tests/benchmark/test_fifo_sizing_flow.py tests/benchmark/test_folding_flow.py tests/benchmark/test_dwc_padding_flow.py"
python -m pytest $T -k "(heuristic_analytical or test_folding_flow or test_padding_flow) and not cybersecurity" \
  -n 8 -q -p no:cacheprovider -rsf > $L/sweep24_A.log 2>&1
echo "EXIT_A $?" >> $L/sweep24_A.log
python -m pytest tests/benchmark/test_fifo_sizing_flow.py -k "(heuristic_rtlsim or largefifo_rtlsim) and not cybersecurity" \
  -n 8 -q -p no:cacheprovider -rsf > $L/sweep24_B.log 2>&1
echo "EXIT_B $?" >> $L/sweep24_B.log
FEATURE_FLOW_QUICK=1 python -m pytest $T -k "cybersecurity and not tightness" -n 6 -q -p no:cacheprovider -rsf > $L/sweep24_C.log 2>&1
echo "EXIT_C $?" >> $L/sweep24_C.log
