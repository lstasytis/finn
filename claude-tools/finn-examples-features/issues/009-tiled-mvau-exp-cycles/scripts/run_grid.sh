#!/bin/bash
# run_grid.sh <grid file> <parallel>: one tiled_sweep.py per line, logs in logs/<tag>.log
S=/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/i009
cd /home/lstasytis/finn/build/wt/fix009
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh . >/dev/null 2>&1
export OMP_NUM_THREADS=2
run_one() {
  tag=$(echo "$@" | tr ' ,' '_x')
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt 209715200 ]; then echo "SKIP(mem) $tag"; return; fi
  python $S/tiled_sweep.py "$@" > $S/logs/$tag.log 2>&1
  grep RESULT $S/logs/$tag.log || echo "FAIL $tag: $(tail -1 $S/logs/$tag.log)"
}
export -f run_one; export S
grep -v '^#' "$1" | xargs -P "$2" -I{} bash -c 'run_one {}'
