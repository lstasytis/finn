"""Regenerate claude-tools/finn-examples-features/STATUS.md from the campaign results.

Sources: the campaign's junit XML + state ($LOGDIR), each feature build's
report/feature_metrics.json, the reference estimate builds, and the hand-written
tools/status_notes.json ({"<row>": "note", "cells": {"<job id>": "override text"}}).
"""
import argparse
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_campaign import (  # noqa: E402
    LOGDIR,
    M,
    TIER1,
    TIER2,
    TIER_TRANSFORMER,
    build_queue,
    junit_result,
)

OUT = os.path.join(M, "claude-tools/finn-examples-features/STATUS.md")
NOTES = os.path.join(M, "claude-tools/finn-examples-features/tools/status_notes.json")
BOARDS = ["AUP-ZU3_8GB", "KV260_SOM", "ZCU104", "U55C", "VCK190", "V80"]
MULTI = {"bnn-pynq", "resnet18", "yolov8", "transformer"}
TAGS = {
    "fifo": "fifo-heuristic_analytical",
    "fold-target": "fold-target",
    "fold-max": "fold-max",
    "dwc": "dwc",
    "fifo-rtlsim": "fifo-heuristic_rtlsim",
    "fold-fifo": "fold-fifoaware",
}
COLS = ["fifo", "fold-target", "fold-max", "dwc", "fifo-rtlsim"]


# row order of the result tables (user): model groups, variants keep their order within a group
MODEL_ORDER = ["tfc", "cnv", "lfc", "gtsrb", "kws", "cybersecurity-mlp", "vgg10-radioml", "mobilenet_v1",
               "resnet50", "resnet18", "espcn", "bert", "yolov8", "transformer"]


def ordered_params(tiers):
    """(mdir, param) pairs of the tiers in MODEL_ORDER (bnn-pynq split into tfc / cnv / lfc)."""
    pairs = [(mdir, p) for mdir, _, params in tiers for p, *_ in params]

    def group(pair):
        mdir, p = pair
        key = p.split("-")[-2] if mdir == "bnn-pynq" else mdir  # e.g. AUP-ZU3_8GB-tfc-w1a1 -> tfc
        return MODEL_ORDER.index(key) if key in MODEL_ORDER else len(MODEL_ORDER)

    return sorted(pairs, key=group)  # stable: variants keep their order


def split_param(p):
    board = next(b for b in BOARDS if p.startswith(b))
    return board, p[len(board) + 1 :] or None


def est_prefix(mdir, param, small=False):
    """Name prefix of a model's cached reference estimate (the tests' reference_estimate)."""
    board, model = split_param(param)
    sm = "small_" if small else ""
    if mdir == "mobilenet_v1":
        return f"est_mobilenet_{sm}v1_{board}_"
    return f"est_{mdir}_{sm}{model}_{board}_" if mdir in MULTI else f"est_{mdir}_{sm}{board}_"


def build_prefix(feature_tag, mdir, param):
    board, model = split_param(param)
    mtag = f"{mdir}_{model}" if mdir in MULTI else mdir
    return f"{feature_tag}_{mtag}_{board}_"


def latest(wt, prefix, rel):
    cands = glob.glob(os.path.join(M, "build/finn_build/fx", wt, prefix + "*", rel))
    return max(cands, key=os.path.getmtime) if cands else None


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return None


def job_metrics(job):
    tag = TAGS[job["feature"]] + ("-small" if job.get("size") == "small" else "")
    prefix = build_prefix(tag, job["model"], job["param"])
    return load(latest(job["wt"], prefix, "report/feature_metrics.json"))


def fifo_aware_cells(job, plain, state, notes):
    """The FIFO-aware folding columns: folding time, throughput, final FIFO size (vs the plain
    optimizer folding at the same target)."""
    head = cell(job, state, notes)
    if not head.startswith(("✅", "❌")):
        return [head, "", ""]
    m, p = job_metrics(job) or {}, (job_metrics(plain) if plain else None) or {}
    t = (m.get("step_times_s") or {}).get("step_target_fps_parallelization")
    tp = (p.get("step_times_s") or {}).get("step_target_fps_parallelization")
    time_s = "%.0f s" % t if t is not None else "-"
    if t is not None and tp:
        time_s += " (plain %.0f s)" % tp
    fps = m.get("rtlsim_fps")
    thr = head[:1] + (" %.1f fps" % fps if fps else "")
    if m.get("rtlsim_vs_est") is not None:
        thr += " (%.3fx est)" % m["rtlsim_vs_est"]
    size = "%.2f kB" % (m["fifo_kbit"] / 8) if m.get("fifo_kbit") is not None else "-"
    if p.get("fifo_kbit"):
        size += " (plain %.2f)" % (p["fifo_kbit"] / 8)
    return [time_s, thr, size]


def pad_compare_cells(job):
    """DWC padding vs none (the DWC test's "pad_compare"): padded / unpadded resources at the
    reference target, padded / unpadded maximize throughput (full size only), padding DWCs of
    the padded target folding; "-" where the job has no comparison (yet)."""
    if job is None:
        return ["-", "-", "-"]
    c = (job_metrics(job) or {}).get("pad_compare")
    if not c:
        return ["-", "-", "-"]
    res = "%.3fx" % c["res_ratio"] + "".join(" +" + r for r in c.get("res_extra") or [])
    res += " (%s, seed %d)" % (c["verdict"], c["seed"])
    mx = c.get("maximize")
    fps = "%.3fx (%s pad DWC)" % (mx["fps_ratio"], mx["pad"]["pad_dwcs"]) if mx else "-"
    return [res, fps, str(c["pad"]["pad_dwcs"])]


def cell(job, state, notes):
    if job["id"] in notes.get("cells", {}):
        return notes["cells"][job["id"]]
    res = junit_result(os.path.join(LOGDIR, job["id"] + ".xml"))
    fin = state.get("finished", {}).get(job["id"], {})
    if job["id"] in state.get("running", {}):
        return "⏳"
    if res is None:
        return "aborted" if fin.get("result") == "aborted" else "⏸"
    if res == "skipped":
        return "skip (no padding)" if job["feature"] == "dwc" else "skip"
    metrics = job_metrics(job)
    parts = ["✅" if res == "passed" else "❌"]
    if metrics:
        if metrics.get("rtlsim_vs_est") is not None:
            parts.append("%.3fx" % metrics["rtlsim_vs_est"])
        if job["feature"].startswith("fold") or job["feature"] == "dwc":
            if metrics.get("est_vs_ref") is not None:
                parts.append("est %.2fx ref" % metrics["est_vs_ref"])
        if metrics.get("fifo_kbit") is not None:
            parts.append("%.2f kB" % (metrics["fifo_kbit"] / 8))
        if job["feature"] == "dwc" and metrics.get("pad_dwcs") is not None:
            parts.append("%d pad DWC" % metrics["pad_dwcs"])
        times = metrics.get("step_times_s") or {}
        key = "step_set_fifo_depths" if job["feature"].startswith("fifo") else (
            "step_target_fps_parallelization")
        if key in times:
            parts.append("%.1f s" % times[key])
    if res != "passed":
        parts.append(notes.get("issues", {}).get(job["id"], "(see log)"))
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="")
    ap.add_argument("--sizes", default=None)
    args = ap.parse_args()
    sizes = args.sizes.split(",") if args.sizes else None
    state = load(os.path.join(LOGDIR, "campaign_state.json")) or {}
    notes = load(NOTES) or {}
    jobs = build_queue(args.variant, sizes)
    tiers = TIER1 + TIER2 + (TIER_TRANSFORMER if args.variant else [])
    lines = [
        "# Feature-test status (models x features)",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `tools/status_table.py` from the "
        "campaign results" + (f" (variant {args.variant})" if args.variant else "") + ". Cells: "
        "result, rtlsim/estimate throughput ratio, (folding/DWC) own estimate vs the committed "
        "folding's estimate, FIFO storage (kB, 1 kB = 1024 bytes), padding DWCs, time of the feature's step (measured under "
        "load). With sizes, a cell lists the small folding first, then the full one. ⏳ running, "
        "⏸ not run yet, blocked = its small run failed. FIFO-aware folding (optional, "
        "`enable_folding_fifo_heuristic`) at the reference target: folding time, throughput, final "
        "FIFO size, each vs the plain optimizer folding. DWC pad/no-pad: the optimizer with padding vs "
        "without at the same seed, estimated resources (mean of the LUT/BRAM_18K/URAM/DSP ratios) at "
        "the reference target (best of the seeds) and estimated throughput in maximize mode.",
        "",
        "| model (board) | est. fps (committed) | FIFO analytical | Folding target | "
        "Folding maximize | DWC | DWC pad/no-pad res. (target) | DWC pad/no-pad fps (maximize) | "
        "DWC #pad DWCs | FIFO-aware: time | FIFO-aware: throughput | "
        "FIFO-aware: FIFO size | FIFO rtlsim | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mdir, p in ordered_params(tiers):
        if True:
            board, model = split_param(p)
            row = f"{model or mdir} ({board})"
            mtag = f"{mdir}_{model}" if mdir in MULTI else mdir
            est = None
            for wt in ("fifo", "folding", "dwc"):
                est = est or load(latest(wt + args.variant, f"est_{mtag}_{board}_",
                                         "report/estimate_network_performance.json"))
            est_s = "%.1f" % est["estimated_throughput_fps"] if est else "-"

            def of(feature, size=None):
                return [j for j in jobs if j["feature"] == feature and j["model"] == mdir
                        and j["param"] == p and (size is None or j.get("size") == size)]

            cells = []
            for col in ["fifo", "fold-target", "fold-max", "dwc"]:
                cells.append(" / ".join(cell(j, state, notes) for j in of(col)) or "-")
            dwc = [pad_compare_cells(j) for j in of("dwc")]
            for i in range(3):
                vals = [d[i] for d in dwc]
                cells.append(" / ".join(vals) if any(v != "-" for v in vals) else "-")
            fifo_aware = of("fold-fifo")
            plain = of("fold-target", "full") or of("fold-target")
            if fifo_aware:
                cells += fifo_aware_cells(fifo_aware[0], plain[0] if plain else None, state, notes)
            else:
                cells += ["-", "-", "-"]
            cells.append(" / ".join(cell(j, state, notes) for j in of("fifo-rtlsim")) or "-")
            lines.append(f"| {row} | {est_s} | " + " | ".join(cells) +
                         f" | {notes.get(row, '')} |")
    lines.append("")
    lines += notes.get("footer", [])
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
