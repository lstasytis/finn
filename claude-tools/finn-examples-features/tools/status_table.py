"""Regenerate claude-tools/finn-examples-features/STATUS.md from the campaign results.

Sources: the campaign's junit XML + state ($LOGDIR), each feature build's
report/feature_metrics.json, the reference estimate builds, and the hand-written
tools/status_notes.json ({"<row>": "note", "cells": {"<job id>": "override text"}}).
"""
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_campaign import LOGDIR, M, TIER1, TIER2, build_queue, junit_result  # noqa: E402

OUT = os.path.join(M, "claude-tools/finn-examples-features/STATUS.md")
NOTES = os.path.join(M, "claude-tools/finn-examples-features/tools/status_notes.json")
BOARDS = ["AUP-ZU3_8GB", "KV260_SOM", "ZCU104", "U55C", "VCK190", "V80"]
MULTI = {"bnn-pynq", "resnet18", "yolov8"}
TAGS = {
    "fifo": "fifo-heuristic_analytical",
    "fold-target": "fold-target",
    "fold-max": "fold-max",
    "dwc": "dwc",
    "fifo-rtlsim": "fifo-heuristic_rtlsim",
}
COLS = ["fifo", "fold-target", "fold-max", "dwc", "fifo-rtlsim"]


def split_param(p):
    board = next(b for b in BOARDS if p.startswith(b))
    return board, p[len(board) + 1 :] or None


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
    metrics = load(latest(job["wt"], build_prefix(TAGS[job["feature"]], job["model"],
                                                  job["param"]), "report/feature_metrics.json"))
    parts = ["✅" if res == "passed" else "❌"]
    if metrics:
        if metrics.get("rtlsim_vs_est") is not None:
            parts.append("%.3fx" % metrics["rtlsim_vs_est"])
        if job["feature"].startswith("fold") or job["feature"] == "dwc":
            if metrics.get("est_vs_ref") is not None:
                parts.append("est %.2fx ref" % metrics["est_vs_ref"])
        if metrics.get("fifo_kbit") is not None:
            parts.append("%.1f kbit" % metrics["fifo_kbit"])
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
    state = load(os.path.join(LOGDIR, "campaign_state.json")) or {}
    notes = load(NOTES) or {}
    jobs = build_queue()
    lines = [
        "# Feature-test status (models x features)",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `tools/status_table.py` from the "
        "campaign results. Cells: result, rtlsim/estimate throughput ratio, (folding/DWC) own "
        "estimate vs the committed folding's estimate, FIFO storage, padding DWCs, time of the "
        "feature's step (FIFO sizing or folding; measured under load, not comparable to isolated "
        "timings). ⏳ running, ⏸ not run yet, aborted = killed at the reboot deadline.",
        "",
        "| model (board) | est. fps (committed) | FIFO analytical | Folding target | "
        "Folding maximize | DWC | FIFO rtlsim | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mdir, stem, params in TIER1 + TIER2:
        for p, *_ in params:
            board, model = split_param(p)
            row = f"{model or mdir} ({board})"
            mtag = f"{mdir}_{model}" if mdir in MULTI else mdir
            est = None
            for wt in ("fifo", "folding", "dwc"):
                est = est or load(latest(wt, f"est_{mtag}_{board}_",
                                         "report/estimate_network_performance.json"))
            est_s = "%.1f" % est["estimated_throughput_fps"] if est else "-"
            cells = []
            for col in COLS:
                job = next(j for j in jobs if j["feature"] == col and j["model"] == mdir
                           and j["param"] == p)
                cells.append(cell(job, state, notes))
            lines.append(f"| {row} | {est_s} | " + " | ".join(cells) +
                         f" | {notes.get(row, '')} |")
    lines.append("")
    lines += notes.get("footer", [])
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
