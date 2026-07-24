#!/usr/bin/env python3
"""Collect REAL LUTs for the DWC plots via Vivado out-of-context synthesis.
HLS variants: vitis_hls csynth (to emit Verilog) -> Vivado synth of that Verilog.
RTL variant: Vivado synth of the dwc_axi core. All numbers are Vivado 'Slice LUTs'
so they are comparable and scale with width (csynth estimates do not)."""
import json, os, re, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ["FINN_ROOT"]
BASE = os.path.join(ROOT, "notebooks", "generalized_dwc")
WORK = tempfile.mkdtemp(prefix="dwc_viv_")

def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

def _slice_luts(util):
    if not os.path.isfile(util):
        return None
    for pat in ("Slice LUTs", "CLB LUTs"):
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
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200)
    return _slice_luts(os.path.join(d, "util.rpt"))

def run_hls(harness, inw, outw):
    d = os.path.join(WORK, f"{harness}_{inw}_{outw}")
    shutil.copytree(os.path.join(BASE, harness), d, dirs_exist_ok=True)
    shutil.rmtree(os.path.join(d, "proj"), ignore_errors=True)
    env = dict(os.environ, INW=str(inw), OUTW=str(outw))
    subprocess.run(["vitis_hls", "-f", "run.tcl"], cwd=d, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1200)
    vdir = os.path.join(d, "proj/sol1/syn/verilog")
    vfiles = [os.path.join(vdir, f) for f in os.listdir(vdir)] if os.path.isdir(vdir) else []
    vfiles = [f for f in vfiles if f.endswith((".v", ".sv"))]
    if not vfiles:
        return None
    return vivado_synth(d, vfiles, "top")

def run_rtl(inw, outw):
    d = os.path.join(WORK, f"rtl_{inw}_{outw}")
    os.makedirs(d, exist_ok=True)
    hdl = os.path.join(ROOT, "finn-rtllib/dwc/hdl")
    return vivado_synth(d, [os.path.join(hdl, "dwc.sv"), os.path.join(hdl, "dwc_axi.sv")],
                        "dwc_axi", {"IBITS": inw, "OBITS": outw})

def job(variant, inw, outw):
    try:
        if variant == "rtl":
            return run_rtl(inw, outw)
        if variant == "hls_mult":
            return run_hls("ref_plain", inw, outw)
        if variant == "hls_nomult":
            if (inw // gcd(inw, outw)) * outw > 8191:
                return None
            return run_hls("ref_lcm", inw, outw)
        return run_hls("ref_gen", inw, outw)
    except Exception as e:
        return f"ERR:{type(e).__name__}"

P1_M = [(20,10),(100,10),(300,10),(500,10),(750,10),(1000,10)]
P1_N = [(25,10),(105,10),(305,10),(505,10),(755,10),(1005,10)]
P2_M = [(10,20),(10,100),(10,300),(10,500),(10,750),(10,1000)]
P2_N = [(10,25),(10,105),(10,305),(10,505),(10,755),(10,1005)]
P3_M = [(64,32),(128,64),(256,128),(512,256),(768,384),(1024,512)]
P3_N = [(44,48),(88,96),(176,192),(352,384),(480,512),(720,768)]
plots = {
    "plot1": {"rtl":P1_M,"hls_mult":P1_M,"gen_mult":P1_M,"hls_nomult":P1_N,"gen_nomult":P1_N},
    "plot2": {"rtl":P2_M,"hls_mult":P2_M,"gen_mult":P2_M,"hls_nomult":P2_N,"gen_nomult":P2_N},
    "plot3": {"rtl":P3_M,"hls_mult":P3_M,"gen_mult":P3_M,"hls_nomult":P3_N,"gen_nomult":P3_N},
}
sv = lambda l: "gen" if l.startswith("gen") else l
tasks = [(p, l, i, o) for p, ls in plots.items() for l, prs in ls.items() for (i, o) in prs]

results = {}
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = {ex.submit(job, sv(l), i, o): (p, l, i, o) for (p, l, i, o) in tasks}
    for f in as_completed(futs):
        p, l, i, o = futs[f]
        lut = f.result()
        results.setdefault(p, {}).setdefault(l, []).append([i, o, lut])
        print(f"{p} {l} {i}->{o}: {lut}", flush=True)
for p in results:
    for l in results[p]:
        results[p][l].sort(key=lambda r: (r[0] + r[1]))
json.dump(results, open(os.path.join(BASE, "plotdata.json"), "w"), indent=1)
print("WROTE plotdata.json (real Vivado LUTs)")
