#!/bin/bash
# Requeue finished campaign jobs (after a fix) and restart the runner: requeue.sh <job id> ...
# The runner adopts its running jobs; requeued jobs run again in queue order.
set -e
L=/home/lstasytis/finn/build/finn_build/fx/logs/campaign
T=/home/lstasytis/finn/claude-tools/finn-examples-features/tools
BUDGET=${BUDGET:-160}
DEADLINE=${DEADLINE:-19:10}
mkdir -p $L/superseded
for j in "$@"; do
  for ext in xml log; do [ -f $L/$j.$ext ] && mv $L/$j.$ext $L/superseded/$j.$(date +%H%M).$ext; done
done
python3 - "$@" <<PY
import json, sys
p = "$L/campaign_state.json"
d = json.load(open(p))
for j in sys.argv[1:]:
    d.get("finished", {}).pop(j, None)
json.dump(d, open(p, "w"), indent=1)
PY
if [ -f $L/runner.pid ] && kill -0 $(cat $L/runner.pid) 2>/dev/null; then
  kill $(cat $L/runner.pid); sleep 3
fi
cd $L && (setsid nohup python3 $T/run_campaign.py --max-threads $BUDGET --deadline $DEADLINE >> runner.log 2>&1 < /dev/null &)
sleep 5; echo "runner pid $(cat $L/runner.pid)"
