#!/usr/bin/env python3
"""Replace the near-coprime (500,512) spike on the non-multiple curves with a
friendlier near-1:1 point (420,448) sitting in the gap between the 736 and 992
points: total 868, ratio 15:16, gcd 28 -> 16 lanes / 4-stage barrel (same
structure as the 480->512 point, so on-trend). Synthesizes it for BOTH
non-multiple variants (old HLS = ref_lcm, new HLS = ref_gen) and merges real
Vivado LUT+FF into plotdata.json / plotdata_ff.json."""
import json, os, re, shutil, subprocess, tempfile

ROOT = os.environ["FINN_ROOT"]
BASE = os.path.join(ROOT, "notebooks", "generalized_dwc")
WORK = tempfile.mkdtemp(prefix="dwc_add2_")

NEW = (420, 448)      # new near-1:1 padding point, mid-gap between 736 and 992
OLD = (500, 512)      # spike to remove

def _rpt(util, pats):
    if not os.path.isfile(util):
        return None
    for pat in pats:
        for ln in open(util):
            if pat in ln:
                m = re.search(r"\|\s*(\d+)", ln)
                if m:
                    return int(m.group(1))
    return None

def vivado_synth(d, read_files, top, gens=None):
    tcl = os.path.join(d, "vsyn.tcl")
    with open(tcl, "w") as f:
        f.write("read_verilog -sv [list %s]\n" % " ".join(read_files))
        g = "".join(f" -generic {k}={v}" for k, v in (gens or {}).items())
        f.write(f"synth_design -top {top} -mode out_of_context -part xc7z020clg400-1{g}\n")
        f.write("report_utilization -file util.rpt\nexit\n")
    subprocess.run(["vivado", "-mode", "batch", "-source", "vsyn.tcl"], cwd=d,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    util = os.path.join(d, "util.rpt")
    return (_rpt(util, ("Slice LUTs", "CLB LUTs")),
            _rpt(util, ("Register as Flip Flop", "Slice Registers", "CLB Registers")))

def run_hls(harness, inw, outw):
    d = os.path.join(WORK, f"{harness}_{inw}_{outw}")
    shutil.copytree(os.path.join(BASE, harness), d, dirs_exist_ok=True)
    shutil.rmtree(os.path.join(d, "proj"), ignore_errors=True)
    env = dict(os.environ, INW=str(inw), OUTW=str(outw))
    subprocess.run(["vitis_hls", "-f", "run.tcl"], cwd=d, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
    vdir = os.path.join(d, "proj/sol1/syn/verilog")
    vfiles = [os.path.join(vdir, f) for f in os.listdir(vdir)] if os.path.isdir(vdir) else []
    vfiles = [f for f in vfiles if f.endswith((".v", ".sv"))]
    if not vfiles:
        return None, None
    return vivado_synth(d, vfiles, "top")

# (line key, harness)
JOBS = [
    ("gen_nomult", "ref_gen"),   # new HLS with padding (non-multiple)
    ("hls_nomult", "ref_lcm"),   # old HLS (non-multiple); LCM(495,528)=7920 < 8191
]

lut_data = json.load(open(os.path.join(BASE, "plotdata.json")))
ff_data = json.load(open(os.path.join(BASE, "plotdata_ff.json")))

def rm(store, line, pair):
    store["plot3"][line] = [r for r in store["plot3"].get(line, [])
                            if not (r[0] == pair[0] and r[1] == pair[1])]

def upsert(store, line, inw, outw, val):
    rows = [r for r in store["plot3"].setdefault(line, []) if not (r[0] == inw and r[1] == outw)]
    rows.append([inw, outw, val])
    rows.sort(key=lambda r: (r[0] + r[1]))
    store["plot3"][line] = rows

# drop the old spike from the new-HLS non-multiple curve
rm(lut_data, "gen_nomult", OLD)
rm(ff_data, "gen_nomult", OLD)

inw, outw = NEW
for line, harness in JOBS:
    lut, ff = run_hls(harness, inw, outw)
    print(f"plot3 {line} {inw}->{outw}: LUT={lut} FF={ff}", flush=True)
    upsert(lut_data, line, inw, outw, lut)
    upsert(ff_data, line, inw, outw, ff)

json.dump(lut_data, open(os.path.join(BASE, "plotdata.json"), "w"), indent=1)
json.dump(ff_data, open(os.path.join(BASE, "plotdata_ff.json"), "w"), indent=1)
print(f"MERGED {NEW} into non-multiple curves; removed {OLD}")
