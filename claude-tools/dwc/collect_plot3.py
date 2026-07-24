#!/usr/bin/env python3
"""Re-collect plot3 with realistic near-1:1 padding-style widths (self-contained).
Non-multiple LCM grows past ap_uint's 8191 ceiling for the largest, so the
LCM-cascade fails there (recorded as null)."""
import json, os, re, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.environ["FINN_ROOT"]
BASE = os.path.join(ROOT, "notebooks", "generalized_dwc")
WORK = tempfile.mkdtemp(prefix="dwc_p3_")

def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

def run_hls(harness, inw, outw):
    d = os.path.join(WORK, f"{harness}_{inw}_{outw}")
    shutil.copytree(os.path.join(BASE, harness), d, dirs_exist_ok=True)
    shutil.rmtree(os.path.join(d, "proj"), ignore_errors=True)
    env = dict(os.environ, INW=str(inw), OUTW=str(outw))
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
            return run_rtl(inw, outw)
        if variant == "hls_mult":
            return run_hls("ref_plain", inw, outw)
        if variant == "hls_nomult":
            if (inw // gcd(inw, outw)) * outw > 8191:
                return None
            return run_hls("ref_lcm", inw, outw)
        return run_hls("ref_gen", inw, outw)  # gen_mult / gen_nomult
    except Exception as e:
        return f"ERR:{type(e).__name__}"

P3_M = [(64,32),(128,64),(256,128),(512,256),(768,384),(1024,512)]
P3_N = [(44,48),(88,96),(176,192),(352,384),(480,512),(720,768)]
lines = {"rtl": P3_M, "hls_mult": P3_M, "gen_mult": P3_M, "hls_nomult": P3_N, "gen_nomult": P3_N}
def sv(line):
    return "gen" if line.startswith("gen") else line

tasks = [(line, i, o) for line, pairs in lines.items() for (i, o) in pairs]
res = {}
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(job, sv(line), i, o): (line, i, o) for (line, i, o) in tasks}
    for f in as_completed(futs):
        line, i, o = futs[f]
        lut = f.result()
        res.setdefault(line, []).append([i, o, lut])
        print(f"plot3 {line} {i}->{o}: {lut}", flush=True)
for line in res:
    res[line].sort(key=lambda r: (r[0] + r[1]))

path = os.path.join(BASE, "plotdata.json")
data = json.load(open(path))
data["plot3"] = res
json.dump(data, open(path, "w"), indent=1)
print("updated plot3 in plotdata.json")
