#!/bin/bash
# run.sh <worktree> <logname> <python args...>
WT=$1; LOG=$2; shift 2
cd $WT && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export PYTHONDONTWRITEBYTECODE=1 NUM_DEFAULT_WORKERS=4
cd /tmp/claude-1205/-home-lstasytis-finn/3f1e2c46-7f88-4375-a7c4-e46cd11c8455/scratchpad/i006
exec python "$@" > logs/$LOG.log 2>&1
