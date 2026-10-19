"""Render the impact tables (Markdown for IMPACT.md, reST for the docs) from impact.json.

    python render_impact.py impact.json out.md out.rst
"""
import json
import statistics
import sys

impact = json.load(open(sys.argv[1]))
md_path, rst_path = sys.argv[2:4]

# rule -> (display name, class, what it corrects / is corrected by)
RULES = {
    "no_input_pacing": ("paced graph input", "minimizing", ""),
    "no_slack": ("slack relaxation", "minimizing", ""),
    "no_pacer_reset": ("slack reset past the pacer", "minimizing", "part of slack relaxation"),
    "no_pacer_tol": ("pacer tolerance", "minimizing", "part of slack relaxation / cap"),
    "no_cap": ("throttled cap", "minimizing", ""),
    "no_frame_split": ("two-frame split", "minimizing", ""),
    "no_fill_only": ("fill frame only on join/branch edges", "minimizing", "resnet50 only"),
    "no_burst_floor": ("burst floor", "protecting", "corrects slack relaxation, paced input"),
    "no_cap_margin": ("cap margin", "protecting", "corrects the throttled cap"),
    "no_causal_pass": ("causal pass (max of two passes)", "protecting", "corrects the non-causal pass"),
    "only_causal_pass": ("non-causal pass (max of two passes)", "protecting", "corrects the causal pass"),
    "no_join_rule": ("join/branch edges never relaxed", "protecting", "corrects slack relaxation, cap"),
    "no_idle_floor": ("idle-window floor (branch edges)", "protecting", "corrects the two-frame split"),
    "no_measured_period": ("measured global period", "protecting", "corrects the caller's estimate"),
}
MODELS = [
    "tfc-w1a1",
    "tfc-w2a2",
    "cybersecurity-mlp",
    "kws",
    "cnv-w1a1",
    "cnv-w2a2",
    "gtsrb",
    "vgg10-radioml",
    "mobilenet_v1",
    "resnet50",
]


def pct(x, signed=True):
    if x is None:
        return "-"
    return ("%+.1f" if signed else "%.1f") % x


# ranking
rank_rows = []
for rule, (name, cls, note) in RULES.items():
    depth, kbit, fps, models = [], [], [], []
    for m in MODELS:
        r = impact.get(m, {}).get("rules", {}).get(rule)
        if r is None:
            continue
        if r["depth"] != impact[m]["full"]["depth"]:
            models.append(m)
            depth.append(r["depth_pct"])
            kbit.append(r["kbit_pct"])
        if r["fps_pct"] is not None:
            fps.append((r["fps_pct"], m))
    worst = min(fps) if fps else None
    rank_rows.append(
        {
            "rule": name,
            "class": cls,
            "note": note,
            "n": len(models),
            "median_depth": statistics.median(depth) if depth else 0.0,
            "max_depth": max(depth, key=abs) if depth else 0.0,
            "median_kbit": statistics.median(kbit) if kbit else 0.0,
            "max_kbit": max(kbit, key=abs) if kbit else 0.0,
            "worst_fps": worst,
            "models": models,
            "fps_models": len(fps),
        }
    )
rank_rows.sort(key=lambda r: (-abs(r["max_kbit"]), -r["n"]))

md = ["## Ranking (rule removed vs full sizer; depth/storage % = change of the total when the rule is removed)\n"]
md.append("| rule | class | models affected | median depth % | max depth % | median kbit % | max kbit % | worst fps % when removed | note |")
md.append("|---|---|---|---|---|---|---|---|---|")
for r in rank_rows:
    md.append(
        "| %s | %s | %d (%s) | %s | %s | %s | %s | %s | %s |"
        % (
            r["rule"],
            r["class"],
            r["n"],
            ", ".join(r["models"]) if r["models"] else "none",
            pct(r["median_depth"]),
            pct(r["max_depth"]),
            pct(r["median_kbit"]),
            pct(r["max_kbit"]),
            ("%s (%s, %d models simulated)" % (pct(r["worst_fps"][0]), r["worst_fps"][1], r["fps_models"]))
            if r["worst_fps"]
            else "-",
            r["note"],
        )
    )

md.append("\n## Per model (rows: rule removed; depth %, storage %, rtlsim fps % relative to the full sizer)\n")
for m in MODELS:
    e = impact.get(m)
    if e is None:
        continue
    full = e["full"]
    ref = e["reference"]
    md.append(
        "### %s: full sizer depth %d, %.1f kbit, %s fps; reference (largefifo_rtlsim) %s fps, %s kbit\n"
        % (
            m,
            full["depth"],
            full["kbit"],
            "%.0f" % full["fps"] if full["fps"] else "-",
            "%.0f" % ref["fps"] if ref else "-",
            "%.1f" % ref["kbit"] if ref else "-",
        )
    )
    rows = [(rule, r) for rule, r in e["rules"].items() if rule in RULES and (abs(r["depth_pct"]) > 0.05 or abs(r["kbit_pct"]) > 0.05 or r["fps"] is not None)]
    if not rows:
        md.append("no rule changes a depth on this model\n")
        continue
    md.append("| rule removed | depth | depth % | kbit | kbit % | fps | fps % |")
    md.append("|---|---|---|---|---|---|---|")
    for rule, r in sorted(rows, key=lambda kv: -abs(kv[1]["kbit_pct"])):
        md.append(
            "| %s | %d | %s | %.1f | %s | %s | %s |"
            % (
                RULES[rule][0],
                r["depth"],
                pct(r["depth_pct"]),
                r["kbit"],
                pct(r["kbit_pct"]),
                "%.0f" % r["fps"] if r["fps"] else "-",
                pct(r["fps_pct"]) if r["fps_pct"] is not None else "-",
            )
        )
    md.append("")
open(md_path, "w").write("\n".join(md) + "\n")

# reST: ranking table + one compact per-model table
def rst_table(header, rows):
    cols = len(header)
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(cols)]
    sep = "  ".join("=" * w for w in widths)
    out = [sep, "  ".join(str(header[i]).ljust(widths[i]) for i in range(cols)), sep]
    for r in rows:
        out.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(cols)))
    out.append(sep)
    return "\n".join(out)


rows = []
for r in rank_rows:
    rows.append(
        (
            r["rule"],
            r["class"],
            "%d" % r["n"],
            pct(r["median_kbit"]),
            pct(r["max_kbit"]),
            pct(r["max_depth"]),
            ("%s (%s)" % (pct(r["worst_fps"][0]), r["worst_fps"][1])) if r["worst_fps"] and r["worst_fps"][0] < -0.05 else ("none (%d sim.)" % r["fps_models"] if r["fps_models"] else "-"),
            ", ".join(r["models"]) if r["models"] else "none",
        )
    )
rst = [
    rst_table(
        ("rule", "class", "models", "median kbit %", "max kbit %", "max depth %", "worst fps %", "models affected"),
        rows,
    )
]
# compact per-model table: rules x models, cell = kbit % / fps %
affected_rules = [r for r in RULES if any(abs(impact.get(m, {}).get("rules", {}).get(r, {"kbit_pct": 0})["kbit_pct"]) > 0.05 for m in MODELS)]
sim_models = [m for m in MODELS if impact.get(m, {}).get("full", {}).get("fps")]
dep_models = [m for m in MODELS if any(abs(impact[m]["rules"].get(r, {"kbit_pct": 0})["kbit_pct"]) > 0.05 for r in RULES)]
header = ("rule removed",) + tuple(dep_models)
rows = []
for rule in affected_rules:
    cells = []
    for m in dep_models:
        r = impact[m]["rules"].get(rule)
        if r is None or (abs(r["depth_pct"]) <= 0.05 and abs(r["kbit_pct"]) <= 0.05):
            cells.append("=")
        else:
            cell = "%s / %s" % (pct(r["depth_pct"]), pct(r["kbit_pct"]))
            if r["fps_pct"] is not None:
                cell += " / %s" % pct(r["fps_pct"])
            cells.append(cell)
    rows.append((RULES[rule][0],) + tuple(cells))
rst.append(rst_table(header, rows))
open(rst_path, "w").write("\n\n".join(rst) + "\n")
print("\n".join(md[:len(rank_rows) + 3]))
