#!/bin/bash
# Timed comparison on the Alveo U55C (100 MHz): FINN's Vitis flow vs the island flow (compute
# kernel built with islands, linked per model with v++). One build at a time; nothing else
# should run on the machine. Frontends must exist in $D/<model> (run_bnn.py / run_vgg10.py /
# run_mobilenet.py --mode frontend). Models: tfc-w1a1, cnv-w1a1, vgg10, mnv1 (MobileNetV1,
# finn-examples U250 folding), mnv1h (half the U250 PE). Island region: FINN_RWI_REGION if set,
# else the per-model default (pm on the U55C).
#
#   MODELS="tfc-w1a1 cnv-w1a1 vgg10 mnv1h" MODES="bitfile islands" experiments/dynarapid/run_u55c_timing.sh
#
# Results: $D/timing/runs.txt (one JSON line per run), per run $D/timing/<model>_<mode>.log.
set -u
: "${FINN_BUILD_DIR:=/home/lstasytis/finn/build/finn_build}"
: "${PLATFORM_REPO_PATHS:=/mnt/labstore/Xilinx/2025.1/Vitis/platforms}"
export FINN_BUILD_DIR PLATFORM_REPO_PATHS
D=${D:-$FINN_BUILD_DIR/rwu}
MODELS=${MODELS:-"tfc-w1a1 cnv-w1a1"}
MODES=${MODES:-"bitfile islands"}
WORKERS=${WORKERS:-64}
export NUM_DEFAULT_WORKERS=$WORKERS
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $D/timing

for m in $MODELS; do
  for mode in $MODES; do
    # output dir name of the mode (run_vgg10.py calls FINN's flow "vivado")
    odir=$mode
    case $m in
      vgg10)
        [ $mode = bitfile ] && odir=vivado
        cmd="python $HERE/run_vgg10.py --model $FINN_ROOT/tests/benchmark/models/radioml_w4a4_small_tidy.onnx --board U55C --clk 10 --out $D/$m --mode $odir" ;;
      mnv1h)
        cmd="python $HERE/run_mobilenet.py --board U55C --clk 10 --folding $HERE/folding_mobilenet_U250_halfpe.json --out $D/$m --mode $mode --workers $WORKERS" ;;
      mnv1)
        cmd="python $HERE/run_mobilenet.py --board U55C --clk 10 --out $D/$m --mode $mode --workers $WORKERS" ;;
      *)
        cmd="python $HERE/run_bnn.py --model $m --board U55C --clk 10 --out $D/$m --mode $mode --workers $WORKERS" ;;
    esac
    rm -rf $D/$m/$odir $D/$m/$odir.json
    log=$D/timing/${m}_${mode}.log
    echo "$(date +%T) start $m $mode"
    t0=$(date +%s)
    /usr/bin/time -v $cmd > $log 2>&1
    rc=$?
    t1=$(date +%s)
    python - $D $m $mode $rc $((t1 - t0)) $odir >> $D/timing/runs.txt <<'EOF'
import glob, json, os, re, sys
D, m, mode, rc, wall, odir = sys.argv[1:]
out = os.path.join(D, m, odir)
res = {"model": m, "mode": mode, "rc": int(rc), "wall_s": int(wall)}
res["xclbin"] = os.path.isfile(os.path.join(out, "bitfile", "finn-accel.xclbin"))
# the v++ link project of this run (the newest one whose config matches the mode)
links = sorted(glob.glob(os.path.join(os.environ["FINN_BUILD_DIR"], "vitis_link_proj_*")), key=os.path.getmtime)
for L in reversed(links):
    cfg = open(os.path.join(L, "config.txt")).read()
    if ("rwislands" in cfg) == (mode == "islands") and os.path.getmtime(L) > os.path.getmtime(out):
        res["link_dir"] = L
        impl = os.path.join(L, "_x/link/vivado/vpl/prj/prj.runs/impl_1")
        rs = glob.glob(os.path.join(impl, "*route_status.rpt"))
        if rs:
            mm = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", open(rs[0]).read())
            res["routing_errors"] = int(mm.group(1)) if mm else None
        ts = os.path.join(impl, "dr_timing_summary.rpt")
        if os.path.isfile(ts):
            t = open(ts).read()
            t = t[t.find("Design Timing Summary"):]
            mm = re.search(r"WNS\(ns\).*?\n[- ]+\n\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)", t, re.S)
            if mm:
                res["wns_ns"], res["whs_ns"] = float(mm.group(1)), float(mm.group(2))
        # v++ link stage durations from its log ("Starting <x>.." / "Finished <n> of 6")
        va = os.path.join(L, "v++_a.log")
        if os.path.isfile(va):
            st = re.findall(r"^\[(\d\d):(\d\d):(\d\d)\] (Starting .*|Finished \d+\w* of \d+)", open(va).read(), re.M)
            res["link_events"] = ["%s:%s:%s %s" % s for s in st]
        break
k = glob.glob(os.path.join(out, "rwislands", "*", "rwislands_kernel.json"))
if k:
    kr = json.load(open(k[0]))
    res["kernel"] = {"status": kr["status"], "total_s": kr.get("total_s"), "stamps": kr.get("stamps"),
                     "islands": len(kr.get("islands", {})), "stitch_unrouted_pins": kr.get("stitch_unrouted_pins")}
print(json.dumps(res))
EOF
    echo "$(date +%T) done $m $mode rc=$rc wall=$((t1 - t0)) s"
  done
done
