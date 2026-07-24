#!/usr/bin/env python3
"""Extract FF (Slice Register / Flip Flop) counts from the Vivado OOC util.rpt
files left behind by collect_vivado.py, mirroring plotdata.json. Writes
plotdata_ff.json (same schema: {plot: {line: [[inw, outw, ff], ...]}})."""
import glob, json, os, re

BASE = os.path.dirname(os.path.abspath(__file__))
# newest sweep dir wins; fall back to any if some runs only exist elsewhere
WORKDIRS = sorted(glob.glob("/tmp/dwc_viv_*"), key=os.path.getmtime, reverse=True)

def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)

def _ff(util):
    if not os.path.isfile(util):
        return None
    for pat in ("Register as Flip Flop", "Slice Registers", "CLB Registers"):
        for ln in open(util):
            if pat in ln:
                m = re.search(r"\|\s*(\d+)", ln)
                if m:
                    return int(m.group(1))
    return None

def find_ff(subdir):
    for w in WORKDIRS:
        ff = _ff(os.path.join(w, subdir, "util.rpt"))
        if ff is not None:
            return ff
    return None

# variant -> run-dir prefix (same mapping collect_vivado.py used)
DIR = {"rtl": "rtl", "hls_mult": "ref_plain", "hls_nomult": "ref_lcm",
       "gen_mult": "ref_gen", "gen_nomult": "ref_gen"}

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

results = {}
for p, lines in plots.items():
    for l, pairs in lines.items():
        for (i, o) in pairs:
            # LCM cascade fails to synth once LCM > 8191 (same guard as LUT sweep)
            if l == "hls_nomult" and (i // gcd(i, o)) * o > 8191:
                ff = None
            else:
                ff = find_ff(f"{DIR[l]}_{i}_{o}")
            results.setdefault(p, {}).setdefault(l, []).append([i, o, ff])
            print(f"{p} {l} {i}->{o}: {ff}", flush=True)
for p in results:
    for l in results[p]:
        results[p][l].sort(key=lambda r: (r[0] + r[1]))
json.dump(results, open(os.path.join(BASE, "plotdata_ff.json"), "w"), indent=1)
print("WROTE plotdata_ff.json (real Vivado FFs)")
