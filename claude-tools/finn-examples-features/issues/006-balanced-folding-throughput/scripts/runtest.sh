#!/bin/bash
# runtest.sh <logname> <pytest node id>
cd /home/lstasytis/finn/build/wt/fix006 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export NUM_DEFAULT_WORKERS=4
exec python -m pytest -p no:cacheprovider -v "$2" > /tmp/claude-1205/-home-lstasytis-finn/3f1e2c46-7f88-4375-a7c4-e46cd11c8455/scratchpad/i006/logs/$1.log 2>&1
