"""Tables and plots of the build-time campaign (run_campaign.sh): FINN's regular Vivado flow
("vivado"), the fair baseline ("global": the island flow's synthesis and cached shell, one global
place and route) and the island flow ("islands", "islands_fast"), on the U55C (Vivado-only shell)
and the ZCU104, per clock.

Only island / global runs of the current flow count: runs.jsonl lines after the line count of a
runs_before_gate_*.jsonl backup next to it (the flow before the result gate, 2026-10-09) are
used; FINN's Vivado flow did not change, its newest run is used. A run is "ok" when the flow's
gate passed: bitstream written, routing complete, setup and hold met (FINN's flow: the same
checks on its reports). Failures are reported, not dropped: every table has the status and
the success rate per flow.

Size: the synthesized accelerator (sum over nodes, island runs) in % of the device; x-axis of the
plots = the binding resource, max(LUT %, DSP %, BRAM %, URAM %).

    python summarize_campaign.py [--out <dir>]
"""

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_breakdown import island_row, vivado_row  # noqa: E402

# (the container default FINN_BUILD_DIR, /tmp, does not hold the campaign: see run_campaign.sh)
B = os.environ.get("RWSCALE_BUILD_DIR", "/home/lstasytis/finn/build/finn_build")
CAMPAIGNS = [
    ("U55C", 10.0, os.path.join(B, "rwscale", "clk10")),
    ("U55C", 5.0, os.path.join(B, "rwscale", "clk5")),
    ("U55C", 3.333, os.path.join(B, "rwscale", "clk3.333")),
    ("ZCU104", 10.0, os.path.join(B, "rwz", "clk10")),
    ("U55C", 10.0, os.path.join(B, "rwr50", "clk10")),
]
DEVICE = {
    "U55C": {"lut": 1303680, "dsp": 9024, "bram": 2016, "uram": 960},
    "ZCU104": {"lut": 230400, "dsp": 1728, "bram": 312, "uram": 96},
}
MODES = ["vivado", "global", "islands", "islands_fast"]
LABEL = {"vivado": "FINN Vivado", "global": "global (fair)", "islands": "islands", "islands_fast": "islands fast"}
MODEL = {"vgg10": "VGG10", "mnv1": "MobileNetV1", "tfc": "TFC", "cnv": "CNV", "cnv1": "CNV PE=SIMD=1", "r50": "ResNet50"}


def load_runs(d):
    """{(point, mode): run} of a campaign dir: newest run per point and flow, island / global
    runs only after the gate (see the module doc)."""
    path = os.path.join(d, "vshell_timing", "runs.jsonl")
    if not os.path.isfile(path):
        return {}
    lines = [json.loads(x) for x in open(path) if x.strip()]
    skip = 0
    for b in glob.glob(os.path.join(d, "vshell_timing", "runs_before_gate_*.jsonl")):
        skip = max(skip, sum(1 for x in open(b) if x.strip()))
    res = {}
    for k, r in enumerate(lines):
        if r["mode"] != "vivado" and k < skip:
            continue
        res[(r["model"], r["mode"])] = r
    return res


def is_ok(r):
    if r["mode"] == "vivado":
        return bool(r.get("rc") == 0 and r.get("bitfile") and r.get("routing_errors") == 0
                    and r.get("wns_ns") is not None and r["wns_ns"] >= 0
                    and r.get("whs_ns") is not None and r["whs_ns"] >= 0)
    return r.get("status") == "ok"


def status(r):
    if is_ok(r):
        return "ok"
    if r["mode"] == "vivado":
        if r.get("rc") != 0 or not r.get("bitfile"):
            return "failed"
        return "timing" if r.get("routing_errors") == 0 else "route"
    s = r.get("status") or "failed"
    det = r.get("islands_detail") or {}
    if any(v.get("timed_out") for v in det.values()):
        return "island timeout"
    return {"timing_failed": "timing", "island_failed": "island failed"}.get(s, s)


def size(d, point):
    """Synthesized accelerator (sum over nodes) of an island or global run of the point."""
    for mode in ("islands", "islands_fast"):
        j = os.path.join(d, "vshell_timing", "%s_%s" % (point, mode), "islands", "rwislands_zynq.json")
        if os.path.isfile(j):
            r = json.load(open(j))
            isl = r.get("islands") or {}
            if isl:
                tot = {}
                for v in isl.values():
                    for k, x in v["res"].items():
                        tot[k] = tot.get(k, 0) + x
                return tot
    return None


def pct(tot, board):
    dev = DEVICE[board]
    p = {k: 100.0 * tot.get(k, 0) / dev[k] for k in dev}
    p["bind"] = max(p.values())
    p["bind_kind"] = max(dev, key=lambda k: p[k])
    return p


def global_row(out_dir, wall):
    """Fair baseline: FINN preparation, node synthesis (parallel), link, open shell, read
    accelerator, opt, place, phys_opt, route (+ hold repair, post-route phys_opt), bitstream,
    reports."""
    r = json.load(open(os.path.join(out_dir, "islands", "rwislands_zynq.json")))
    st, g = r["stamps"], r.get("global_stamps", {})
    seq = ["link", "open_shell", "read_accel", "opt", "place", "phys_opt", "route", "hold_repair",
           "post_route_phys_opt", "bitstream", "reports"]
    prev, d = 0.0, {}
    for k in seq:
        if k in g:
            d[k] = g[k] - prev
            prev = g[k]
    route = sum(d.get(k, 0) for k in ("route", "hold_repair", "post_route_phys_opt"))
    return [wall - r["total_s"], st.get("synth"), (d.get("link") or 0) + (d.get("open_shell") or 0) + (d.get("read_accel") or 0),
            d.get("opt"), d.get("place"), d.get("phys_opt"), route, d.get("bitstream"), d.get("reports"), wall]


GLOBAL_COLS = ["FINN prep", "node synth (parallel)", "link + open shell + read", "opt", "place", "phys_opt",
               "route (+hold, post-route)", "bitstream", "reports", "total"]
ISLAND_COLS = ["FINN prep", "node synth (parallel)", "floorplan", "island P&R (parallel)", "stitch",
               "asm: read accel", "asm: route", "asm: bitstream", "asm: reports", "total"]
VIVADO_COLS = ["FINN prep (HLS, stitched IPs)", "BD + OOC IP synth", "top synth", "opt", "place", "phys_opt",
               "route", "bitstream", "other (reports, ckpts)", "total"]


def f0(v):
    return "-" if v is None else "%.0f" % v


def collect():
    out = []
    for board, clk, d in CAMPAIGNS:
        runs = load_runs(d)
        if not runs:
            continue
        points = sorted({p for p, _ in runs}, key=lambda p: (p.split("_")[0], _factor(p)))
        rows = []
        for p in points:
            sz = size(d, p)
            row = {"point": p, "size": pct(sz, board) if sz else None, "runs": {}}
            for m in MODES:
                r = runs.get((p, m))
                if r is None:
                    continue
                out_dir = os.path.join(d, "vshell_timing", "%s_%s" % (p, m))
                try:
                    if m == "vivado":
                        br = vivado_row(out_dir, r["wall_s"])
                    elif m == "global":
                        br = global_row(out_dir, r["wall_s"])
                    else:
                        br = island_row(out_dir, r["wall_s"])[0]
                except Exception:
                    br = None
                row["runs"][m] = {"wall": r["wall_s"], "ok": is_ok(r), "status": status(r), "br": br,
                                  "wns": r.get("wns_ns"), "whs": r.get("whs_ns"), "islands": r.get("islands")}
            rows.append(row)
        out.append({"board": board, "clk": clk, "dir": d, "rows": rows})
    return out


def _factor(p):
    m = re.search(r"_s([\d.]+)", p)
    return float(m.group(1)) if m else 1.0


def pretty(p):
    m = re.match(r"(\w+?)_(s([\d.]+)([rn])|h)$", p)
    if not m:
        return p
    name = MODEL.get(m.group(1), m.group(1))
    if m.group(2) == "h":
        return "%s (hand-written folding)" % name
    return "%s %sx%s" % (name, m.group(3), " (no relax)" if m.group(4) == "n" else "")


def tables(camps):
    md = []
    for c in camps:
        mhz = round(1000 / c["clk"])
        md += ["### %s, %d MHz" % (c["board"], mhz), ""]
        hdr = ["point", "binding res. %"] + ["%s s" % LABEL[m] for m in MODES] + [
            "islands vs FINN", "islands vs global", "islands fast vs FINN"]
        md += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        succ = {m: [0, 0] for m in MODES}
        for row in c["rows"]:
            cells = [pretty(row["point"]),
                     "%.0f (%s)" % (row["size"]["bind"], row["size"]["bind_kind"]) if row["size"] else "-"]
            for m in MODES:
                r = row["runs"].get(m)
                if r is None:
                    cells.append("")
                    continue
                succ[m][1] += 1
                succ[m][0] += r["ok"]
                cells.append(f0(r["wall"]) + ("" if r["ok"] else " (%s)" % r["status"]))

            def sp(a, b):
                ra, rb = row["runs"].get(a), row["runs"].get(b)
                if ra and rb and ra["ok"] and rb["ok"]:
                    return "%.2fx" % (rb["wall"] / ra["wall"])
                return "-"

            cells += [sp("islands", "vivado"), sp("islands", "global"), sp("islands_fast", "vivado")]
            md.append("| " + " | ".join(cells) + " |")
        md += ["", "Success (bitstream, routed, setup and hold met): " + ", ".join(
            "%s %d/%d" % (LABEL[m], s[0], s[1]) for m, s in succ.items() if s[1]), ""]
        for m, cols in (("vivado", VIVADO_COLS), ("global", GLOBAL_COLS), ("islands", ISLAND_COLS), ("islands_fast", ISLAND_COLS)):
            rs = [(row["point"], row["runs"][m]) for row in c["rows"] if m in row["runs"] and row["runs"][m]["br"]]
            if not rs:
                continue
            md += ["#### %s, %d MHz: %s breakdown (s)" % (c["board"], mhz, LABEL[m]), ""]
            md += ["| point | " + " | ".join(cols) + " | status |", "|" + "---|" * (len(cols) + 2)]
            for p, r in rs:
                md.append("| %s | %s | %s |" % (pretty(p), " | ".join(f0(v) for v in r["br"]), r["status"]))
            md.append("")
    return "\n".join(md) + "\n"


def plot(camps, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"vgg10": "#2a78d6", "mnv1": "#eb6834", "r50": "#1a9e77", "tfc": "#7a5195", "cnv": "#ef5675", "cnv1": "#ffa600"}
    dash = {"vivado": (0, (5, 3)), "global": (0, (1, 2)), "islands": "solid", "islands_fast": (0, (8, 2, 2, 2))}
    marker = {"vivado": "s", "global": "^", "islands": "o", "islands_fast": "D"}
    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e6e5e1", "#fcfcfb"
    paths = []
    for c in camps:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.set_facecolor(surface)
        for model in color:
            for m in MODES:
                pts = sorted(
                    (row["size"]["bind"], row["runs"][m]["wall"], row["runs"][m]["ok"])
                    for row in c["rows"]
                    if row["point"].split("_")[0] == model and m in row["runs"] and row["size"]
                )
                if not pts:
                    continue
                xs, ys, oks = zip(*pts)
                ax.plot(xs, ys, linestyle=dash[m], linewidth=1.8, color=color[model],
                        label="%s, %s" % (MODEL[model], LABEL[m]), zorder=2)
                ax.scatter(xs, ys, s=50, marker=marker[m], color=[color[model] if ok else surface for ok in oks],
                           edgecolors=color[model], linewidths=1.8, zorder=3)
        ax.set_title("%s, %d MHz" % (c["board"], round(1000 / c["clk"])), color=ink, fontsize=11, loc="left")
        ax.set_xlabel("accelerator size: binding resource, % of the device", color=muted, fontsize=9)
        ax.set_ylabel("build time, model to bitstream (s)", color=muted, fontsize=9)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=muted, labelsize=9)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.text(0.01, 0.01, "hollow marker: failed, or routing / setup / hold not met", color=muted, fontsize=7)
        fig.patch.set_facecolor(surface)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        p = os.path.join(out, "campaign_%s_%dMHz.png" % (c["board"], round(1000 / c["clk"])))
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(B, "campaign"))
    a = ap.parse_args()
    camps = collect()
    md = "# Build-time campaign: FINN Vivado vs fair global baseline vs island flow\n\n" + tables(camps)
    p = os.path.join(a.out, "campaign_tables.md")
    open(p, "w").write(md)
    json.dump(camps, open(os.path.join(a.out, "campaign.json"), "w"), indent=1)
    print(md)
    print("plots:", plot(camps, a.out))


if __name__ == "__main__":
    main()
