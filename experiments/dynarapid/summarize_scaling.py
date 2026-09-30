"""Summarize the core-scaling runs of run_scaling.sh (Phase 4).

For each run <model>_<mode>_n<N> in <dir>: wall time of the bitfile flow, stage breakdown
(DynaRapid: synthesis done / batches done / library done, stitching, assembly), CPU work
(user + system time of the whole process tree, from /usr/bin/time -v), peak system memory in
use during the run (sampled with free every 5 s, minus the idle level at its start), WNS and
routing errors. Writes <dir>/scaling.json and prints a table.

Usage: python summarize_scaling.py <scaling dir>
"""

import json
import os
import re
import sys


def _time_v(log):
    txt = open(log, errors="ignore").read()
    r = {}
    for key, pat in (
        ("user_s", r"User time \(seconds\): ([\d.]+)"),
        ("sys_s", r"System time \(seconds\): ([\d.]+)"),
        ("max_rss_gb", r"Maximum resident set size \(kbytes\): (\d+)"),
    ):
        m = re.search(pat, txt)
        if m:
            r[key] = float(m.group(1))
    if "max_rss_gb" in r:
        r["max_rss_gb"] /= 2**20
    return r


def _mem(f):
    if not os.path.isfile(f):
        return None
    vals = [int(line.split()[1]) for line in open(f) if len(line.split()) == 2]
    return (max(vals) - vals[0]) / 1024 if vals else None


def summarize(d):
    rows = []
    for name in sorted(os.listdir(os.path.join(d, "bit"))):
        m = re.match(r"(\w+?)_(vivado|dynarapid)_n(\d+)$", name)
        if not m:
            continue
        model, mode, n = m.group(1), m.group(2), int(m.group(3))
        r = {"model": model, "mode": mode, "cores": n}
        bj = os.path.join(d, "bit", name, "bitfile_experiment.json")
        if os.path.isfile(bj):
            b = json.load(open(bj))
            r["total_s"] = b.get("total_s")
            r["wns_ns"] = b.get("wns_ns")
        zj = os.path.join(d, "bit", name, "dynarapid", "dynarapid_zynq.json")
        if os.path.isfile(zj):
            z = json.load(open(zj))
            r.update(
                status=z.get("status"),
                routing_errors=z.get("nets_with_routing_errors"),
                library_s=z.get("parallel_s"),
                stitch_s=z.get("stitch_s"),
                assembly_s=z.get("assembly_s"),
            )
        log = os.path.join(d, name + ".log")
        if os.path.isfile(log):
            r.update(_time_v(log))
            lib = re.search(r"DynaRapid library: (\{[^}]*\})", open(log, errors="ignore").read())
            if lib:
                st = json.loads(lib.group(1).replace("'", '"'))
                r["synth_done_s"] = st.get("synth_done_s")
                r["batches_done_s"] = st.get("batches_done_s")
                r["fallback"] = st.get("fallback")
                # components per batched Vivado run (library plan; fixed ~5-6 before it)
                r["batch_k"] = st.get("k")
        r["peak_mem_gb"] = _mem(os.path.join(d, "mem_%s.txt" % name))
        if "user_s" in r and r.get("total_s"):
            r["cpu_s"] = r["user_s"] + r.get("sys_s", 0)
            r["avg_cores"] = r["cpu_s"] / r["total_s"]
        rows.append(r)
    return rows


def main():
    d = sys.argv[1]
    rows = summarize(d)
    with open(os.path.join(d, "scaling.json"), "w") as f:
        json.dump(rows, f, indent=2)
    cols = [
        "model", "mode", "cores", "total_s", "library_s", "synth_done_s", "stitch_s",
        "assembly_s", "cpu_s", "avg_cores", "peak_mem_gb", "wns_ns", "routing_errors",
        "fallback", "batch_k",
    ]
    print(" | ".join(cols))
    for r in sorted(rows, key=lambda r: (r["model"], r["mode"], -r["cores"])):
        out = []
        for c in cols:
            v = r.get(c)
            out.append("%.1f" % v if isinstance(v, float) else str(v if v is not None else "-"))
        print(" | ".join(out))


if __name__ == "__main__":
    main()
