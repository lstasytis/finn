"""Table of the U55C timing runs (run_u55c_timing.sh): per model and flow the wall time, the v++
link stage durations, routing errors, WNS/WHS and (islands) the kernel build breakdown.

Usage: python summarize_u55c.py <rwu dir>   (e.g. $FINN_BUILD_DIR/rwu)
"""

import datetime
import glob
import json
import os
import re
import sys

STAGES = [
    ("Starting logic optimization", "opt"),
    ("Starting logic placement", "place"),
    ("Starting logic routing", "route"),
    ("Starting bitstream generation", "bitstream"),
]


def secs(hms):
    h, m, s = [int(x) for x in hms.split(":")]
    return h * 3600 + m * 60 + s


def link_info(L):
    res = {"link_dir": os.path.basename(L)}
    va = os.path.join(L, "v++_a.log")
    txt = open(va, errors="ignore").read() if os.path.isfile(va) else ""
    # vpl steps (synth = platform IP OOC synthesis, impl = everything after)
    steps = re.findall(r"^\[(\d\d:\d\d:\d\d)\] Run vpl: Step (\w+): (Started|Completed)", txt, re.M)
    t = {}
    for hms, step, what in steps:
        t.setdefault(step, {})[what] = secs(hms)
    for step in ("synth", "impl"):
        if step in t and "Started" in t[step] and "Completed" in t[step]:
            res[step + "_s"] = (t[step]["Completed"] - t[step]["Started"]) % 86400
    ev = [(secs(h), e) for h, e in re.findall(r"^\[(\d\d:\d\d:\d\d)\] (Starting [^.]+|Finished 6th)", txt, re.M)]
    marks = {}
    for ts, e in ev:
        for key, name in STAGES:
            if e.startswith(key):
                marks[name] = ts
        if e.startswith("Finished 6th"):
            marks["end"] = ts
    order = ["opt", "place", "route", "bitstream", "end"]
    for a, b in zip(order, order[1:]):
        if a in marks and b in marks:
            res[a + "_s"] = (marks[b] - marks[a]) % 86400
    impl = os.path.join(L, "_x/link/vivado/vpl/prj/prj.runs/impl_1")
    rs = glob.glob(os.path.join(impl, "*route_status.rpt"))
    if rs:
        m = re.search(r"# of nets with routing errors\.+ :\s+(\d+)", open(rs[0]).read())
        res["routing_errors"] = int(m.group(1)) if m else None
    ts = os.path.join(impl, "dr_timing_summary.rpt")
    if os.path.isfile(ts):
        tt = open(ts).read()
        tt = tt[tt.find("Design Timing Summary") :]
        m = re.search(r"WNS\(ns\).*?\n[- ]+\n\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)", tt, re.S)
        if m:
            res["wns"], res["whs"] = m.group(1), m.group(2)
    runme = os.path.join(impl, "runme.log")
    if os.path.isfile(runme):
        m = re.search(r"RWI_HOOK done \S+ ([\d.]+)", open(runme, errors="ignore").read())
        if m:
            res["hook_s"] = float(m.group(1))
    return res


def main():
    d = sys.argv[1]
    log = open(os.path.join(d, "timing_u55c.log")).read()
    starts = {(m, mo): t for t, m, mo in re.findall(r"^(\d\d:\d\d:\d\d) start (\S+) (\S+)", log, re.M)}
    dones = {(m, mo): (t, int(w)) for t, m, mo, w in re.findall(r"^(\d\d:\d\d:\d\d) done (\S+) (\S+) rc=\d+ wall=(\d+)", log, re.M)}
    links = glob.glob(os.path.join(os.environ.get("FINN_BUILD_DIR", os.path.dirname(d)), "vitis_link_proj_*"))
    day = datetime.date.today()
    rows = []
    for key, (t_end, wall) in dones.items():
        m, mode = key
        t0 = datetime.datetime.combine(day, datetime.time(*[int(x) for x in starts[key].split(":")])).timestamp()
        t1 = datetime.datetime.combine(day, datetime.time(*[int(x) for x in t_end.split(":")])).timestamp()
        cand = []
        for L in links:
            c = os.stat(L).st_ctime
            cfg = open(os.path.join(L, "config.txt")).read() if os.path.isfile(os.path.join(L, "config.txt")) else ""
            if t0 <= c <= t1 and ("rwislands" in cfg) == (mode == "islands") and m in cfg + L:
                cand.append(L)
            elif t0 <= c <= t1 and ("rwislands" in cfg) == (mode == "islands"):
                cand.append(L)
        row = {"model": m, "mode": mode, "wall_s": wall}
        if cand:
            row.update(link_info(sorted(cand)[-1]))
        k = glob.glob(os.path.join(d, m, mode, "rwislands", "*", "rwislands_kernel.json"))
        if k:
            kr = json.load(open(k[0]))
            st = kr.get("stamps", {})
            row["kernel"] = {
                "total_s": round(kr.get("total_s", 0)),
                "synth_s": round(st.get("synth", 0)),
                "islands_pnr_s": round(st.get("islands", 0) - st.get("floorplan", 0)),
                "stitch_s": round(st.get("stitch", 0) - st.get("islands", 0)),
                "K": len(kr.get("islands", {})),
            }
        rows.append(row)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
