#!/bin/bash
# After the cnv-w1a1 variants: gtsrb + cnv-w2a2 (reference, full, variants), then vgg10 + mobilenet.
source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh
S=$(dirname $(dirname "$0")); T=$S/tools
cd $FINN_ROOT
J=/tmp/claude-1205/-home-lstasytis-finn-dev-lstasytis-finn/8871d419-69de-49fc-824c-30ba6c67e801/tasks/bjisghkzg.output
until grep -q "RESULT fifo_abl_no_causal_pass cnv-w1a1\|run FAILED\|config FAILED" $J 2>/dev/null; do sleep 60; done
echo "=== group 2: gtsrb, cnv-w2a2 $(date)"
FEATURE_FLOW_MODELS=gtsrb,cnv-w2a2 FEATURE_FLOW_BOARDS=ZCU104 timeout 20000 python -m pytest tests/benchmark/test_fifo_sizing_flow.py -k "largefifo_rtlsim or heuristic_analytical" -n 4 -p no:cacheprovider -q --tb=line -W ignore 2>&1 | grep "passed\|failed" | tail -2
for v in no_burst_floor no_frame_split no_input_pacing no_slack no_causal_pass; do $T/throughput.sh gtsrb - $v & $T/throughput.sh bnn-pynq cnv-w2a2 $v & wait; done
$T/cleanup.sh
echo "=== group 3: vgg10-radioml, mobilenet_v1 $(date)"
FEATURE_FLOW_MODELS=vgg10-radioml,mobilenet_v1 FEATURE_FLOW_BOARDS=ZCU104 timeout 30000 python -m pytest tests/benchmark/test_fifo_sizing_flow.py -k "largefifo_rtlsim or heuristic_analytical" -n 4 -p no:cacheprovider -q --tb=line -W ignore 2>&1 | grep "passed\|failed" | tail -2
for v in no_burst_floor no_slack no_pacer_tol no_pacer_reset no_frame_split; do $T/throughput.sh vgg10-radioml - $v; done
for v in no_burst_floor no_cap no_slack no_pacer_tol no_pacer_reset no_frame_split no_input_pacing no_causal_pass; do $T/throughput.sh mobilenet_v1 - $v; $T/cleanup.sh; done
echo "=== chain done $(date)"
