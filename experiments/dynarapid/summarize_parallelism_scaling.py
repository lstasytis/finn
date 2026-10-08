"""Tables and plots of the parallelism-scaling experiment (run_parallelism_scaling.sh).

Per clock ($D/clk<ns>): one row per model point and flow with the synthesized accelerator size
(sum of the island flow's synthesized nodes: LUT, DSP, BRAM36, URAM, as % of the U55C) and the
wall-time breakdown in the same columns for both flows; one plot per clock with build time over
size, 4 lines (VGG10 / MobileNet x global Vivado / island flow).

x-axis (--x): "area" = (LUT% + DSP%) / 2 (DSP counted as the device's LUT/DSP ratio, ~144 LUTs),
"max" = max(LUT%, DSP%) (the binding compute resource), "lut", "dsp".

    python summarize_parallelism_scaling.py [--x area] [--out <dir>]
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_breakdown import island_row, vivado_row  # noqa: E402

D = os.environ.get("D") or os.path.join(
    os.environ.get("FINN_BUILD_DIR", "/home/lstasytis/finn/build/finn_build"), "rwscale"
)
# xcu55c-fsvh2892-2L-e
DEVICE = {"lut": 1303680, "dsp": 9024, "bram": 2016, "uram": 960}
MODELS = {"vgg10": "VGG10", "mnv1": "MobileNet"}
COLS = [
    "FINN prep",
    "synthesis",
    "P&R before stitching",
    "stitch",
    "read accel",
    "final route",
    "bitstream",
    "other",
    "total",
]


def runs(path):
    if not os.path.isfile(path):
        return {}
    res = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            res[(r["model"], r["mode"])] = r  # the last run of a point and flow
    return res


def point_key(name):
    """vgg10_s1.25n -> ("vgg10", 1.25, "n")"""
    m = re.match(r"(\w+?)_s([\d.]+)([rn])$", name)
    return m.group(1), float(m.group(2)), m.group(3)


def size(isl_json):
    r = json.load(open(isl_json))
    tot = dict.fromkeys(DEVICE, 0.0)
    for isl in r.get("islands", {}).values():
        for k in tot:
            tot[k] += isl["res"].get(k, 0)
    pct = {k: 100.0 * tot[k] / DEVICE[k] for k in DEVICE}
    pct["area"] = (pct["lut"] + pct["dsp"]) / 2
    pct["max"] = max(pct["lut"], pct["dsp"])
    return tot, pct, len(r.get("islands", {}))


def common_row(flow, vals):
    """The flows' own breakdowns (summarize_breakdown) in common columns."""
    if flow == "islands":
        prep, synth, fp, isl, stitch, read, route, bit, rep, total = vals
        return [prep, synth, (fp or 0) + (isl or 0), stitch, read, route, bit, rep, total]
    prep, ooc, top, opt, place, phys, route, bit, other, total = vals
    return [prep, (ooc or 0) + (top or 0), (opt or 0) + (place or 0) + (phys or 0), None, None, route, bit, other, total]


def fmt(v, d=0):
    return "-" if v is None else ("%%.%df" % d) % v


def collect(clk_dir):
    T = os.path.join(clk_dir, "vshell_timing")
    rr = runs(os.path.join(T, "runs.jsonl"))
    rows = []
    for (name, flow), r in rr.items():
        try:
            model, factor, mode = point_key(name)
        except AttributeError:
            continue
        out = os.path.join(T, "%s_%s" % (name, flow))
        isl_json = os.path.join(T, name + "_islands", "islands", "rwislands_zynq.json")
        sz = size(isl_json) if os.path.isfile(isl_json) else None
        # a bitstream with 0 routing errors that meets setup and hold
        ok = (r.get("rc") == 0 and r.get("bitfile") and r.get("routing_errors") in (0, None)
              and (r.get("wns_ns") or 0) >= 0 and (r.get("whs_ns") or 0) >= 0)
        try:
            vals = island_row(out, r["wall_s"])[0] if flow == "islands" else vivado_row(out, r["wall_s"])
            br = common_row(flow, vals)
        except Exception:
            br = [None] * (len(COLS) - 1) + [r.get("wall_s")]
        rows.append(dict(model=model, factor=factor, mode=mode, flow=flow, ok=bool(ok), size=sz, br=br,
                         wns=r.get("wns_ns"), whs=r.get("whs_ns"), islands=r.get("islands")))
    rows.sort(key=lambda x: (x["model"], x["factor"], x["flow"] != "vivado"))
    return rows


def table(rows, clk_ns):
    out = ["#### %g MHz (clock %s ns)" % (round(1000 / clk_ns), clk_ns), ""]
    hdr = ["model", "point", "flow", "LUT %", "DSP %", "BRAM %", "x (area %)", "islands"] + COLS + ["WNS ns", "ok"]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for x in rows:
        sz = x["size"][1] if x["size"] else {}
        pt = "%gx%s" % (x["factor"], " (no relax)" if x["mode"] == "n" else "")
        out.append("| " + " | ".join(
            [MODELS.get(x["model"], x["model"]), pt, "islands" if x["flow"] == "islands" else "Vivado",
             fmt(sz.get("lut"), 1), fmt(sz.get("dsp"), 1), fmt(sz.get("bram"), 1), fmt(sz.get("area"), 1),
             str(x["islands"]) if x["flow"] == "islands" and x["islands"] else "-"]
            + [fmt(v) for v in x["br"]] + [fmt(x["wns"], 3), "yes" if x["ok"] else "**no**"]) + " |")
    return "\n".join(out) + "\n"


def plot(per_clock, xkey, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reference palette (dataviz skill, categorical slots 1-2); hue = model, dash = flow
    color = {"vgg10": "#2a78d6", "mnv1": "#eb6834"}
    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e6e5e1", "#fcfcfb"
    clocks = sorted(per_clock, key=lambda c: -c)
    fig, axes = plt.subplots(1, len(clocks), figsize=(5.2 * len(clocks), 4.2), sharey=True, squeeze=False)
    xlabel = {"area": "accelerator size: (LUT % + DSP %) / 2 of the U55C", "max": "max(LUT %, DSP %) of the U55C",
              "lut": "LUT % of the U55C", "dsp": "DSP % of the U55C"}[xkey]
    for ax, clk in zip(axes[0], clocks):
        ax.set_facecolor(surface)
        for model in ("vgg10", "mnv1"):
            for flow, dash, mk in (("vivado", (0, (5, 3)), "s"), ("islands", "solid", "o")):
                pts = sorted(
                    (x["size"][1][xkey], x["br"][-1], x["ok"])
                    for x in per_clock[clk]
                    if x["model"] == model and x["flow"] == flow and x["size"] and x["br"][-1]
                )
                if not pts:
                    continue
                xs, ys, oks = zip(*pts)
                label = "%s, %s" % (MODELS[model], "island flow" if flow == "islands" else "global Vivado")
                ax.plot(xs, ys, linestyle=dash, linewidth=2, color=color[model], label=label, zorder=2)
                ax.scatter(xs, ys, s=64, marker=mk, color=[color[model] if ok else surface for ok in oks],
                           edgecolors=color[model], linewidths=2, zorder=3)
        ax.set_title("%g MHz" % round(1000 / clk), color=ink, fontsize=11, loc="left")
        ax.set_xlabel(xlabel, color=muted, fontsize=9)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=9)
    axes[0][0].set_ylabel("build time, model to bitstream (s)", color=muted, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.text(0.01, 0.01, "hollow marker: build failed or did not meet timing / route", color=muted, fontsize=8)
    fig.patch.set_facecolor(surface)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default="area", choices=["area", "max", "lut", "dsp"])
    ap.add_argument("--out", default=D)
    a = ap.parse_args()
    per_clock = {}
    for d in sorted(os.listdir(D)) if os.path.isdir(D) else []:
        m = re.match(r"clk([\d.]+)$", d)
        if m:
            per_clock[float(m.group(1))] = collect(os.path.join(D, d))
    md = ["# Parallelism scaling: build time vs accelerator size (U55C, Vivado-only shell)", ""]
    for clk in sorted(per_clock, key=lambda c: -c):
        md.append(table(per_clock[clk], clk))
    md.append(
        "Size: synthesized accelerator (island flow's per-node synthesis), % of the xcu55c. "
        "P&R before stitching: global Vivado opt + place + phys_opt; island flow floorplan + parallel "
        "island place-and-route (incl. the islands' own routing). Final route: global route_design / "
        "the assembly's route. Other: link, reports, checkpoints (Vivado), reports (islands)."
    )
    path = os.path.join(a.out, "scaling_tables.md")
    open(path, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if per_clock:
        png = os.path.join(a.out, "scaling_%s.png" % a.x)
        plot(per_clock, a.x, png)
        print("plot:", png)


if __name__ == "__main__":
    main()
