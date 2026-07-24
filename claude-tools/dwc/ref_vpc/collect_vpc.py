#!/usr/bin/env python3
"""Synthesize the vpc (Vector Pack Converter) RTL for every (inW,outW) point in
the DWC resource benchmark and emit LUT/FF, so vpc can be added as a line to
dwc_resource_analysis.tex alongside rtl / hls / generalized.

Bit-width -> vpc mapping: W=1, PI=inWidth, PO=outWidth, N=lcm(PI,PO) (one exact
repacking frame). vpc normalizes internally by gcd(PI,PO); at W=1 this yields the
same minimal normalized datapath as W=gcd(inW,outW), i.e. the fairest realization.

Runs Vivado OOC synth in parallel and writes vpc_results.json:
  { "plot1": {"vpc_mult":[[inW,outW,LUT,FF],...], "vpc_nomult":[...]}, ... }
"""
import os, re, json, math, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
FINN_ROOT = os.path.abspath(os.path.join(BASE, "..", "..", ".."))
VIV = os.environ.get("VIVADO", "/mnt/labstore/Xilinx/Vivado/2023.1/bin/vivado")
TCL = os.path.join(BASE, "synth.tcl")
VPC_SRC = os.environ.get("VPC_SRC", os.path.join(BASE, "vpc.sv"))  # self-contained copy
WORK = os.path.join(BASE, "work")
JOBS = int(os.environ.get("VPC_JOBS", "8"))

# Same (inW,outW) points as the other variants in plotdata.json.
POINTS = {
    "plot1": {
        "vpc_mult":   [(20,10),(100,10),(300,10),(500,10),(750,10),(1000,10)],
        "vpc_nomult": [(25,10),(105,10),(305,10),(505,10),(755,10),(1005,10)],
    },
    "plot2": {
        "vpc_mult":   [(10,20),(10,100),(10,300),(10,500),(10,750),(10,1000)],
        "vpc_nomult": [(10,25),(10,105),(10,305),(10,505),(10,755),(10,1005)],
    },
    "plot3": {
        "vpc_mult":   [(64,32),(128,64),(256,128),(512,256),(682,341),(768,384),(1024,512)],
        "vpc_nomult": [(44,48),(88,96),(176,192),(352,384),(420,448),(480,512),(720,768)],
    },
}


def lcm(a, b):
    return a * b // math.gcd(a, b)


def synth(inW, outW):
    W, PI, PO = 1, inW, outW
    N = lcm(PI, PO)
    wd = os.path.join(WORK, f"{inW}_{outW}")
    os.makedirs(wd, exist_ok=True)
    env = dict(os.environ, FINN_ROOT=FINN_ROOT, VPC_SRC=VPC_SRC, W=str(W), N=str(N),
               PI=str(PI), PO=str(PO))
    try:
        r = subprocess.run([VIV, "-mode", "batch", "-source", TCL,
                            "-nojournal", "-nolog"], cwd=wd, env=env,
                           capture_output=True, text=True, timeout=1800)
        m = re.search(r"VPC_RESULT LUT=(\d+) FF=(\d+)", r.stdout)
        if not m:
            tail = (r.stdout + r.stderr)[-500:]
            return inW, outW, None, None, f"noparse: {tail}"
        return inW, outW, int(m.group(1)), int(m.group(2)), None
    except subprocess.TimeoutExpired:
        return inW, outW, None, None, "timeout"


def main():
    os.makedirs(WORK, exist_ok=True)
    tasks = {}  # (inW,outW) -> None
    for plot in POINTS.values():
        for pts in plot.values():
            for p in pts:
                tasks[p] = None
    print(f"synthesizing {len(tasks)} unique points, {JOBS} parallel")
    results = {}
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futs = {ex.submit(synth, i, o): (i, o) for (i, o) in tasks}
        for f in as_completed(futs):
            inW, outW, lut, ff, err = f.result()
            results[(inW, outW)] = (lut, ff)
            print(f"  {inW}->{outW}: LUT={lut} FF={ff}" + (f"  ERR={err}" if err else ""), flush=True)

    out = {}
    for plot, lines in POINTS.items():
        out[plot] = {}
        for key, pts in lines.items():
            rows = []
            for (i, o) in pts:
                lut, ff = results.get((i, o), (None, None))
                rows.append([i, o, lut, ff])
            out[plot][key] = rows
    with open(os.path.join(BASE, "vpc_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote vpc_results.json")


if __name__ == "__main__":
    main()
