#!/bin/bash
# Timed comparison: FINN's Vivado ZynqBuild vs the RapidWright island flow (ZCU104, 100 MHz).
# Runs one build at a time (nothing else should run on the machine). Each island run is cold
# (fresh output dir: synthesis, island P&R and stitching from scratch); the 100 MHz shell is
# cached (built once beforehand, its one-time cost reported separately).
#
#   MODELS="tfc cnv cnv1 vgg10 mnv1" MODES="vivado islands" experiments/dynarapid/run_islands_timing.sh
#
# Model inputs (prepared beforehand, see NOTES_ISLANDS.md): $D/<model>/... ; results in
# $D/timing/<model>_<mode>[_<tag>]/bitfile_experiment.json and a line per run in $D/timing/runs.txt.
set -u
: "${FINN_BUILD_DIR:=/home/lstasytis/finn/build/finn_build}"
export FINN_BUILD_DIR
D=${D:-$FINN_BUILD_DIR/rwi}
MODELS=${MODELS:-"tfc cnv cnv1 vgg10 mnv1"}
MODES=${MODES:-"vivado islands"}
WORKERS=${WORKERS:-$(nproc)}
ISLANDS=${ISLANDS:-auto}
TAG=${TAG:-}
SHELLS=$FINN_BUILD_DIR/rwislands/shells
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $D/timing

model_path() {
  case $1 in
    tfc|cnv) echo $D/$1/dataflow_ipgen.onnx ;;
    cnv1|vgg10|mnv1) echo $D/$1/frontend/intermediate_models/step_set_fifo_depths.onnx ;;
  esac
}

for m in $MODELS; do
  for mode in $MODES; do
    out=$D/timing/${m}_${mode}${TAG:+_$TAG}
    rm -rf $out
    mkdir -p $out
    echo "$(date +%T) start $m $mode -> $out"
    /usr/bin/time -v python $HERE/run_bitfile_experiment.py --model $(model_path $m) --out $out \
      --mode $mode --clk 10 --workers $WORKERS --islands $ISLANDS --shell-lib $SHELLS \
      > $out/run.log 2>&1
    rc=$?
    python - $out $m $mode $rc >> $D/timing/runs.txt <<'EOF'
import json, re, sys
out, m, mode, rc = sys.argv[1:]
try:
    r = json.load(open(out + "/bitfile_experiment.json"))
except Exception:
    r = {}
z = r.get("dynarapid_zynq", {})
log = open(out + "/run.log", errors="ignore").read()
mem = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
print(json.dumps({"model": m, "mode": mode, "rc": int(rc), "total_s": r.get("total_s"),
                  "wns_ns": r.get("wns_ns"), "status": z.get("status", "vivado"),
                  "islands": len(z.get("islands", {})), "stamps": z.get("stamps"),
                  "routing_errors": z.get("nets_with_routing_errors"),
                  "driver_rss_gb": int(mem.group(1)) / 2**20 if mem else None}))
EOF
    echo "$(date +%T) done $m $mode rc=$rc: $(tail -n 1 $D/timing/runs.txt)"
  done
done
