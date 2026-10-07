#!/bin/bash
# Timed island-flow builds on an Alveo part with the Vivado-only shell (no Vitis platform;
# finn.util.dynarapid.shell.ALVEO_SHELL), one build at a time, from scratch except the shell
# (cached per board/clock/IODMA interface: build it once with a first, untimed run of each model).
# Frontends must exist in $D/<model>/frontend (run_bnn.py / run_vgg10.py / run_mobilenet.py
# --board U55C --mode frontend).
#
#   MODELS="tfc-w1a1 vgg10 mnv1" [MODE=islands|vivado] experiments/dynarapid/run_vshell_timing.sh
#
# MODE=vivado: FINN's regular Vivado flow on the same shell (one global implementation of the
# shell block design with the stitched IPs, Vivado default settings as v++) - the baseline.
#
# Results: $D/vshell_timing/runs.jsonl (one JSON line per run), logs in $D/vshell_timing/.
set -u
: "${FINN_BUILD_DIR:=/home/lstasytis/finn/build/finn_build}"
export FINN_BUILD_DIR
D=${D:-$FINN_BUILD_DIR/rwu}
MODELS=${MODELS:-"tfc-w1a1 vgg10 mnv1"}
BOARD=${BOARD:-U55C}
MODE=${MODE:-islands}
WORKERS=${WORKERS:-64}
SHELLS=${SHELLS:-$FINN_BUILD_DIR/rwislands/vshells}
HERE=$(cd "$(dirname "$0")" && pwd)
T=$D/vshell_timing
mkdir -p $T

for m in $MODELS; do
  out=$T/${m}_$MODE
  rm -rf $out
  log=$T/${m}_$MODE.log
  echo "$(date +%T) start $m"
  t0=$(date +%s)
  /usr/bin/time -v python $HERE/run_bitfile_experiment.py \
    --model $D/$m/frontend/intermediate_models/step_set_fifo_depths.onnx --out $out \
    --mode $MODE --board $BOARD --clk 10 --workers $WORKERS --shell-lib $SHELLS > $log 2>&1
  rc=$?
  t1=$(date +%s)
  python - $out $m $rc $((t1 - t0)) $MODE >> $T/runs.jsonl <<'EOF'
import glob, json, os, re, sys
out, m, rc, wall, mode = sys.argv[1:]
res = {"model": m, "mode": mode, "rc": int(rc), "wall_s": int(wall)}
if mode == "vivado":
    # the global Vivado project (path in the experiment result)
    e = os.path.join(out, "bitfile_experiment.json")
    proj = json.load(open(e)).get("project") if os.path.isfile(e) else None
    if proj:
        for k, f in (("route", "route_status.rpt"), ("timing", "timing_summary.rpt")):
            res[k + "_rpt"] = os.path.join(proj, f)
        rs = os.path.join(proj, "route_status.rpt")
        if os.path.isfile(rs):
            mm = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", open(rs).read())
            res["routing_errors"] = int(mm.group(1)) if mm else None
        ts = os.path.join(proj, "timing_summary.rpt")
        if os.path.isfile(ts):
            t = open(ts).read()
            t = t[t.find("Design Timing Summary"):]
            mm = re.search(r"WNS\(ns\).*?\n[- ]+\n\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s+(\S+)", t, re.S)
            if mm:
                res["wns_ns"], res["whs_ns"], res["hold_failing"] = float(mm.group(1)), float(mm.group(2)), int(mm.group(3))
        res["bitfile"] = os.path.isfile(os.path.join(proj, "resizer.bit"))
j = os.path.join(out, "islands", "rwislands_zynq.json")
if os.path.isfile(j):
    r = json.load(open(j))
    res.update(status=r["status"], shell=r["shell"].get("status"), islands=len(r.get("islands", {})),
               stamps={k: round(v) for k, v in r.get("stamps", {}).items()},
               assembly={k: round(v) for k, v in r.get("assembly_stamps", {}).items()})
    asm = os.path.join(out, "islands", "assembly")
    rs = os.path.join(asm, "route_status.rpt")
    if os.path.isfile(rs):
        mm = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", open(rs).read())
        res["routing_errors"] = int(mm.group(1)) if mm else None
    ts = os.path.join(asm, "timing_summary.rpt")
    if os.path.isfile(ts):
        t = open(ts).read()
        t = t[t.find("Design Timing Summary"):]
        mm = re.search(r"WNS\(ns\).*?\n[- ]+\n\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+\S+\s+(\S+)", t, re.S)
        if mm:
            res["wns_ns"], res["whs_ns"], res["hold_failing"] = float(mm.group(1)), float(mm.group(2)), int(mm.group(3))
    res["bitfile"] = os.path.isfile(os.path.join(out, "islands", "resizer.bit"))
print(json.dumps(res))
EOF
  echo "$(date +%T) done $m rc=$rc wall=$((t1 - t0)) s"
done
