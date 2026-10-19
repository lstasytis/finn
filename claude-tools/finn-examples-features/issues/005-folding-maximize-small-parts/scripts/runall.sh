#!/bin/bash
# runall.sh <worktree whose src to import> <tag> [jobs]  -- maximize probes on the ZU3 set (+KV260, U55C)
WT=$1; TAG=$2; JOBS=${3:-11}
S=/tmp/claude-1205/-home-lstasytis-finn/3f1e2c46-7f88-4375-a7c4-e46cd11c8455/scratchpad/s005
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh /home/lstasytis/finn/build/wt/fix005 >/dev/null 2>&1
export PYTHONPATH=$WT/src:$WT/tests:$PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $S/$TAG
RUNS=${RUNS:-"bnn-pynq_:cnv-w1a1:AUP-ZU3_8GB bnn-pynq_:cnv-w1a2:AUP-ZU3_8GB bnn-pynq_:cnv-w2a2:AUP-ZU3_8GB bnn-pynq_:lfc-w1a1:AUP-ZU3_8GB bnn-pynq_:lfc-w1a2:AUP-ZU3_8GB bnn-pynq_:tfc-w1a1:AUP-ZU3_8GB bnn-pynq_:tfc-w1a2:AUP-ZU3_8GB bnn-pynq_:tfc-w2a2:AUP-ZU3_8GB :cybersecurity-mlp:AUP-ZU3_8GB :gtsrb:AUP-ZU3_8GB :kws:AUP-ZU3_8GB bnn-pynq_:cnv-w1a2:KV260_SOM bnn-pynq_:cnv-w2a2:U55C"}
for r in $RUNS; do echo $r; done | xargs -P $JOBS -I{} bash -c '
IFS=: read P M B <<< "{}"; W=1024; [ "$M" = cybersecurity-mlp ] && W=80
cd '$S/$TAG' && timeout 2400 taskset -c '${CPUS:-64-79}' python '$S'/probe.py $M $B --prefix "$P" --wwidth $W --out ${M}__${B}.json $EXTRA > ${M}__${B}.log 2>&1; echo done $M $B'
