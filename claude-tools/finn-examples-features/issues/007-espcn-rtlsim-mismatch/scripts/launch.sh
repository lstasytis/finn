#!/bin/bash
# usage: launch.sh <name> <first> <last> [fifo overrides]   (env SUBSIM_* passed through)
S=/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/s
name=$1; shift
avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
if [ "$avail" -lt 200 ]; then echo "MemAvailable ${avail} GB < 200 GB, refusing"; exit 1; fi
cd /home/lstasytis/finn/build/wt/fix007 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix007
export NUM_DEFAULT_WORKERS=2 OMP_NUM_THREADS=2
setsid nohup bash -c "echo \$\$ > $S/$name.pid; exec python $S/subsim.py $* $S/$name" > $S/$name.log 2>&1 < /dev/null &
sleep 2; echo "launched $name pid $(cat $S/$name.pid)"
