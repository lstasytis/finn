#!/bin/bash
# Requeue finished campaign jobs (after a fix) and restart the runner: requeue.sh <job id> ...
# The runner adopts its running jobs; requeued jobs run again in queue order.
set -e
L=/home/lstasytis/finn/build/finn_build/fx/logs/campaign
T=/home/lstasytis/finn/claude-tools/finn-examples-features/tools
BUDGET=${BUDGET:-160}
DEADLINE=${DEADLINE:-none}
mkdir -p $L/superseded
# one requeue at a time (watchers of several agents call this)
exec 9>$L/requeue.lock
flock 9
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
LIVE=()
if [ -f $L/runner.pid ] && kill -0 $(cat $L/runner.pid) 2>/dev/null; then
  RPID=$(cat $L/runner.pid)
  # the live runner's exact arguments (python3 run_campaign.py ARGS...)
  mapfile -d '' LIVE < /proc/$RPID/cmdline
  kill $RPID; sleep 3
fi
if [ ${#LIVE[@]} -gt 2 ]; then ARGS=("${LIVE[@]:2}")
  # keep runner.args current (used when no runner is alive)
  printf '%s ' "${ARGS[@]}" | sed 's/ $/\n/' > $L/runner.args
# else the arguments in runner.args (space-separated, no shell quoting needed)
elif [ -f $L/runner.args ]; then mapfile -t ARGS < <(tr ' ' '\n' < $L/runner.args | grep -v '^$')
else ARGS=(--max-threads $BUDGET --deadline $DEADLINE); fi
cd $L && (setsid nohup python3 $T/run_campaign.py "${ARGS[@]}" >> runner.log 2>&1 < /dev/null 9>&- &)
sleep 5; echo "runner pid $(cat $L/runner.pid)"
