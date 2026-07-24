#!/usr/bin/env python3
"""Synthesize a few extra DWC points (~1024 total bits) and merge their real
Vivado LUT+FF into plotdata.json / plotdata_ff.json under plot3."""
import json, os, re, shutil, subprocess, tempfile

ROOT = os.environ["FINN_ROOT"]
BASE = os.path.join(ROOT, "notebooks", "generalized_dwc")
WORK = tempfile.mkdtemp(prefix="dwc_add_")

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
    lut = _rpt(util, ("Slice LUTs", "CLB LUTs"))
    ff = _rpt(util, ("Register as Flip Flop", "Slice Registers", "CLB Registers"))
    return lut, ff

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

def run_rtl(inw, outw):
    d = os.path.join(WORK, f"rtl_{inw}_{outw}")
    os.makedirs(d, exist_ok=True)
    hdl = os.path.join(ROOT, "finn-rtllib/dwc/hdl")
    return vivado_synth(d, [os.path.join(hdl, "dwc.sv"), os.path.join(hdl, "dwc_axi.sv")],
                        "dwc_axi", {"IBITS": inw, "OBITS": outw})

# (line key, inw, outw, runner)
JOBS = [
    ("rtl",        682, 341, lambda: run_rtl(682, 341)),
    ("hls_mult",   682, 341, lambda: run_hls("ref_plain", 682, 341)),
    ("gen_mult",   682, 341, lambda: run_hls("ref_gen", 682, 341)),
    ("gen_nomult", 500, 512, lambda: run_hls("ref_gen", 500, 512)),
    # hls_nomult (500,512): LCM=64000 > 8191, does not synthesize -> skip (missing point)
]

lut_data = json.load(open(os.path.join(BASE, "plotdata.json")))
ff_data = json.load(open(os.path.join(BASE, "plotdata_ff.json")))

def upsert(store, line, inw, outw, val):
    rows = store["plot3"].setdefault(line, [])
    rows = [r for r in rows if not (r[0] == inw and r[1] == outw)]
    rows.append([inw, outw, val])
    rows.sort(key=lambda r: (r[0] + r[1]))
    store["plot3"][line] = rows

for line, inw, outw, fn in JOBS:
    lut, ff = fn()
    print(f"plot3 {line} {inw}->{outw}: LUT={lut} FF={ff}", flush=True)
    upsert(lut_data, line, inw, outw, lut)
    upsert(ff_data, line, inw, outw, ff)

json.dump(lut_data, open(os.path.join(BASE, "plotdata.json"), "w"), indent=1)
json.dump(ff_data, open(os.path.join(BASE, "plotdata_ff.json"), "w"), indent=1)
print("MERGED extra ~1024-bit points into plotdata.json / plotdata_ff.json")
