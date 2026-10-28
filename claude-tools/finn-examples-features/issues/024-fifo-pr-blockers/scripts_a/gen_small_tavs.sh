#!/bin/bash
# Issue 024 A: TAV models of the newest passing small analytical build per model (fx/fifo2), 3 at a time.
F=/home/lstasytis/finn/build/finn_build/fx/fifo2; O=/home/lstasytis/finn/build/finn_build/fx/fix024a/tav_small; mkdir -p $O
cd /home/lstasytis/finn/build/wt/fix024a
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=$O NUM_DEFAULT_WORKERS=4
cd $O
for key in bnn-pynq_cnv-w1a1_AUP bnn-pynq_cnv-w1a2_AUP bnn-pynq_cnv-w1a2_KV260 bnn-pynq_cnv-w2a2_U55C bnn-pynq_lfc-w1a1 bnn-pynq_lfc-w1a2 bnn-pynq_tfc-w1a1 bnn-pynq_tfc-w1a2 bnn-pynq_tfc-w2a2 cybersecurity-mlp gtsrb kws vgg10-radioml transformer_transformer_benchmark resnet18_resnet18_4w4a resnet50_V80 bert_VCK190; do
  b=$(ls -td $F/fifo-heuristic_analytical-small_${key}* | while read d; do [ -e $d/verification_output/verify_stitched_ip_rtlsim_0_SUCCESS.npy ] && echo $d && break; done)
  [ -z "$b" ] && continue
  while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 10; done
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  [ -s $O/$key.onnx ] || python /home/lstasytis/finn/claude-tools/finn-examples-features/issues/024-fifo-pr-blockers/scripts_a/tav_model.py $b $O/$key.onnx > $O/$key.log 2>&1 &
done; wait; echo DONE > $O/DONE
