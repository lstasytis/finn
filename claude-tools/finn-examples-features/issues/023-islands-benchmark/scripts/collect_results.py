#!/usr/bin/env python3
"""RESULTS table of the island-flow benchmark from the jobs' feature_metrics.json.

  python3 collect_results.py [FINN_BUILD_DIR] > table.md      (default fx/islands2)

Per (model, board, size): the latest island job and the latest baseline job (by start time of
their output dirs); speedup = baseline time / island time (island time without a one-time shell
build). Also writes results.json next to this script's issue folder if --json is given.
"""

import glob
import json
import os
import sys

BD = "/home/lstasytis/finn/build/finn_build/fx/islands2"


def load(fn):
    try:
        with open(fn) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def fmt_t(m):
    if not m or m.get("time_s") is None:
        return "-"
    return "%.0f s%s" % (m["time_s"], "*" if m.get("shell_in_job") else "")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    bd = args[0] if args else BD
    rows = {}
    for fn in glob.glob(os.path.join(bd, "islands-*", "report", "feature_metrics.json")):
        m = load(fn)
        if not m:
            continue
        flow = m["test"].split("-", 1)[1]
        st = m.get("stamps") or {}
        if flow == "islands" and "shell" in st and "synth" in st:
            # time with a cached shell (early runs subtracted the whole shell build)
            built = m.get("shell_status") == "built" or (m.get("shell_lock_wait_s") or 0) > 1
            m["time_s"] = m["total_s"] - (max(0.0, st["shell"] - st["synth"]) if built else 0.0)
            m["shell_in_job"] = built
        key = (m["tag"], m["board"], m.get("model_board"))
        cur = rows.setdefault(key, {})
        if flow not in cur or os.path.getmtime(fn) > cur[flow]["_mtime"]:
            m["_mtime"] = os.path.getmtime(fn)
            cur[flow] = m
    print("| model | size | board (folding of) | clk ns | threads | islands | baseline | speedup "
          "| islands WNS/WHS | baseline WNS | islands | verify | load start/end |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    out = []
    for (tag, board, mb), r in sorted(
        rows.items(), key=lambda kv: (kv[0][0].endswith("_small") is False, kv[0][0], kv[0][1:])
    ):
        isl, base = r.get("islands"), r.get("baseline")
        m = isl or base
        size = "small" if tag.endswith("_small") else "full"
        model = tag[: -len("_small")] if size == "small" else tag
        sp = "-"
        if isl and base and isl.get("status") == "ok" and base.get("status") == "ok":
            sp = "%.2fx" % (base["time_s"] / isl["time_s"])
        t = (isl or {}).get("timing") or {}
        tb = (base or {}).get("timing") or {}
        v = (isl or {}).get("verify") or {}
        verify = "-"
        if isl:
            if v.get("golden_match"):
                verify = "golden match (%d fr)" % v.get("frames", 0)
            elif v:
                verify = "FAIL"
        status = isl.get("status") if isl else "-"
        loads = "%s / %s" % (
            "/".join("%.0f" % x["load_start"] for x in (isl, base) if x),
            "/".join("%.0f" % x["load_end"] for x in (isl, base) if x and x.get("load_end") is not None),
        )
        print("| %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s / %s | %s | %s | %s | %s |" % (
            model, size, m["board"], m.get("model_board"), m.get("clk_ns"), m.get("threads"),
            fmt_t(isl), fmt_t(base), sp, t.get("wns"), t.get("whs"), tb.get("wns"),
            "%s, %s isl." % (status, (isl or {}).get("islands", "-")), verify, loads))
        out.append(dict(tag=tag, board=board, islands=isl, baseline=base, speedup=sp))
    if "--json" in sys.argv:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results.json"), "w") as f:
            json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
