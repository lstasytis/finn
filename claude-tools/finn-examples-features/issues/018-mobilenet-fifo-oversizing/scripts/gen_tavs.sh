#!/bin/bash
# Issue 018: save TAV models (013's tav_model.py) of the mobilenet full/small analytical builds.
# usage: gen_tavs.sh   (PID in $O/gen.pid)
cd /home/lstasytis/finn/build/wt/fix018
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix018
export NUM_DEFAULT_WORKERS=8
S=/home/lstasytis/finn/claude-tools/finn-examples-features/issues/013-fifo-latency-model/scripts
O=$FINN_BUILD_DIR/tav; F=/home/lstasytis/finn/build/finn_build/fx/fifo2
echo $$ > $O/gen.pid
for kb in full:fifo-heuristic_analytical_mobilenet_v1_U55C_anmrhapg small:fifo-heuristic_analytical-small_mobilenet_v1_U55C_qo_bs6s9; do
  k=${kb%%:*}; b=${kb#*:}
  while [ $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) -lt 200 ]; do sleep 30; done
  ( cd $O && python $S/tav_model.py $F/$b $O/mnv1_$k.onnx > $O/mnv1_$k.log 2>&1 )
  echo "$k rc=$? $(tail -1 $O/mnv1_$k.log)" >> $O/gen.log
done
echo DONE >> $O/gen.log
