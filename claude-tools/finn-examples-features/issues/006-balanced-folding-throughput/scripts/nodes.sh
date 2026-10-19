#!/bin/bash
# nodes.sh <tag> <model> <part> <nodes...>
TAG=$1; M=$2; export PART=$3; shift 3
for n in "$@"; do /tmp/claude-1205/-home-lstasytis-finn/3f1e2c46-7f88-4375-a7c4-e46cd11c8455/scratchpad/i006/run.sh /home/lstasytis/finn/build/wt/fix006-diag ns_${TAG}_$n /tmp/claude-1205/-home-lstasytis-finn/3f1e2c46-7f88-4375-a7c4-e46cd11c8455/scratchpad/i006/node_sim.py $M $n 4; done
