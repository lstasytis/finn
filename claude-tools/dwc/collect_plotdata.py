#!/usr/bin/env python3
"""Collect LUT data for the DWC resource-analysis plots.
Variants: rtl (Vivado OOC), hls_mult (StreamingDataWidthConverter_Batch),
hls_nomult (LCM cascade), gen (generalized, both mult and non-mult).
Runs each synth in an isolated temp dir; parallel with a thread pool.
Writes plotdata.json."""
import json, math, os, re, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ["FINN_ROOT"]
BASE = os.path.join(ROOT, "notebooks", "generalized_dwc")
WORK = tempfile.mkdtemp(prefix="dwc_plot_")

def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

def run_hls(harness, inw, outw):
    d = os.path.join(WORK, f"{harness}_{inw}_{outw}")
    src = os.path.join(BASE, harness)
    shutil.copytree(src, d, dirs_exist_ok=True)
    env = dict(os.environ, INW=str(inw), OUTW=str(outw))
    for j in ("proj",):
        shutil.rmtree(os.path.join(d, j), ignore_errors=True)
    subprocess.run(["vitis_hls", "-f", "run.tcl"], cwd=d, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
    xml = os.path.join(d, "proj/sol1/syn/report/top_csynth.xml")
    if not os.path.isfile(xml):
        return None
    m = re.search(r"<LUT>(\d+)</LUT>", open(xml).read())
    return int(m.group(1)) if m else None

def run_rtl(inw, outw):
    d = os.path.join(WORK, f"rtl_{inw}_{outw}")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(BASE, "ref_rtl", "synth_axi.tcl"), d)
    env = dict(os.environ, IBITS=str(inw), OBITS=str(outw))
    p = subprocess.run(["vivado", "-mode", "batch", "-source", "synth_axi.tcl"],
                       cwd=d, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=900, text=True)
    m = re.findall(r"RTL_LUT=(\d+)", p.stdout)
    return int(m[-1]) if m else None

def job(variant, inw, outw):
    try:
        if variant == "rtl":
            lut = run_rtl(inw, outw)
        elif variant == "hls_mult":
            lut = run_hls("ref_plain", inw, outw)
        elif variant == "hls_nomult":
            if (inw // gcd(inw, outw)) * outw > 8191:
                return (variant, inw, outw, None)  # LCM exceeds ap_uint max
            lut = run_hls("ref_lcm", inw, outw)
        elif variant == "gen":
            lut = run_hls("ref_gen", inw, outw)
        return (variant, inw, outw, lut)
    except Exception as e:
        return (variant, inw, outw, f"ERR:{type(e).__name__}")

# ---- width pairs per plot ----
P1_M = [(20,10),(100,10),(300,10),(500,10),(750,10),(1000,10)]
P1_N = [(25,10),(105,10),(305,10),(505,10),(755,10),(1005,10)]
P2_M = [(10,20),(10,100),(10,300),(10,500),(10,750),(10,1000)]
P2_N = [(10,25),(10,105),(10,305),(10,505),(10,755),(10,1005)]
P3_M = [(10,5),(100,50),(200,100),(400,200),(700,350),(1000,500)]
P3_N = [(9,6),(90,60),(180,120),(360,240),(630,420),(900,600)]

plots = {
    "plot1": {"rtl": P1_M, "hls_mult": P1_M, "gen_mult": P1_M, "hls_nomult": P1_N, "gen_nomult": P1_N},
    "plot2": {"rtl": P2_M, "hls_mult": P2_M, "gen_mult": P2_M, "hls_nomult": P2_N, "gen_nomult": P2_N},
    "plot3": {"rtl": P3_M, "hls_mult": P3_M, "gen_mult": P3_M, "hls_nomult": P3_N, "gen_nomult": P3_N},
}
# variant used for synth (gen_mult/gen_nomult both use the "gen" harness)
def synth_variant(line):
    return "gen" if line.startswith("gen") else line

tasks = []
for plot, lines in plots.items():
    for line, pairs in lines.items():
        for (inw, outw) in pairs:
            tasks.append((plot, line, inw, outw))

results = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(job, synth_variant(line), inw, outw): (plot, line, inw, outw)
            for (plot, line, inw, outw) in tasks}
    for f in as_completed(futs):
        plot, line, inw, outw = futs[f]
        _, _, _, lut = f.result()
        results.setdefault(plot, {}).setdefault(line, []).append([inw, outw, lut])
        print(f"{plot} {line} {inw}->{outw}: {lut}", flush=True)

for plot in results:
    for line in results[plot]:
        results[plot][line].sort(key=lambda r: (r[0] + r[1]))
json.dump(results, open(os.path.join(BASE, "plotdata.json"), "w"), indent=1)
print("WROTE plotdata.json")
