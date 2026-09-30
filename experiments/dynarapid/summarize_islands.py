"""Table of the island-flow timing runs (run_islands_timing.sh): per model the Vivado ZynqBuild
time vs the island flow, with the island flow's stage breakdown.

Usage: python summarize_islands.py <timing dir>   (e.g. $FINN_BUILD_DIR/rwi/timing)
"""

import json
import os
import sys


def main():
    d = sys.argv[1]
    runs = {}
    for line in open(os.path.join(d, "runs.txt")):
        r = json.loads(line)
        runs[(r["model"], r["mode"])] = r
    models = []
    for m, _ in runs:
        if m not in models:
            models.append(m)
    print(
        "| model | Vivado s | WNS | islands s | speedup | K | synth | island P&R | stitch |"
        " assembly | WNS | routing errors |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in models:
        v, i = runs.get((m, "vivado"), {}), runs.get((m, "islands"), {})
        st = i.get("stamps") or {}
        prep = None
        # the island flow's own stamps start after ZynqBuild's partitioning / IODMA HLS
        if i.get("total_s") and st.get("assembly"):
            prep = i["total_s"] - st["assembly"]

        def f(x, n=0):
            return "-" if x is None else ("%.*f" % (n, x))

        speed = v.get("total_s") / i["total_s"] if v.get("total_s") and i.get("total_s") else None
        print(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                m,
                f(v.get("total_s")),
                f(v.get("wns_ns"), 3),
                f(i.get("total_s")),
                f(speed, 2) + "x" if speed else "-",
                i.get("islands", "-"),
                f(st.get("synth")),
                f(st.get("islands", 0) - st.get("floorplan", 0) if st else None),
                f(st.get("stitch", 0) - st.get("islands", 0) if st else None),
                f(st.get("assembly", 0) - st.get("stitch", 0) if st else None),
                f(i.get("wns_ns"), 3),
                i.get("routing_errors", "-"),
            )
        )
        if prep is not None:
            print("|  (islands: ZynqBuild prep before the flow %.0f s) |||||||||||" % prep)


if __name__ == "__main__":
    main()
