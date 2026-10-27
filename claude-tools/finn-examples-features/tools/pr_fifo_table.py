"""Markdown result table of the FIFO sizing feature tests for the PR description (tooling only).

    python pr_fifo_table.py [--notes notes.json] > table.md

One row per model and folding size (small, full): heuristic_analytical's stitched-IP rtlsim throughput
over the build's own estimate, its total FIFO storage and sizing time, and the same sizer with rtlsim
token access vectors (heuristic_rtlsim) for comparison. notes.json: {"<job id>": "text for the result
cell"} replaces the result of a job (e.g. the cause of a failure that is not a FIFO matter).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_campaign import LOGDIR, TIER1, TIER2, TIER_TRANSFORMER, build_queue, junit_result  # noqa: E402
from status_table import job_metrics, load, ordered_params, split_param  # noqa: E402


def fmt_time(s):
    return "%.1f s" % s if s < 100 else "%.0f s" % s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()
    notes = load(args.notes) if args.notes else {}
    state = load(os.path.join(LOGDIR, "campaign_state.json")) or {}
    jobs = build_queue("2", ["small", "full"])

    def job(mdir, p, feature, size):
        return next((j for j in jobs if j["model"] == mdir and j["param"] == p
                     and j["feature"] == feature and j.get("size") == size), None)

    def outcome(j):
        if j is None:
            return None, {}
        if j["id"] in state.get("running", {}):
            return "running", {}
        return junit_result(os.path.join(LOGDIR, j["id"] + ".xml")), job_metrics(j) or {}

    print("| model | board | folding | rtlsim / estimate | FIFO storage | sizing time | sizing time, rtlsim TAVs "
          "| result |")
    print("|---|---|---|---:|---:|---:|---:|---|")
    for mdir, p in ordered_params(TIER1 + TIER2 + TIER_TRANSFORMER):
        board, model = split_param(p)
        for size in ("small", "full"):
            ja, jr = job(mdir, p, "fifo", size), job(mdir, p, "fifo-rtlsim", size)
            res, m = outcome(ja)
            _, mr = outcome(jr)
            if res is None and (ja is None or ja["id"] not in notes):
                continue
            t = (m.get("step_times_s") or {}).get("step_set_fifo_depths")
            tr = (mr.get("step_times_s") or {}).get("step_set_fifo_depths")
            ratio = m.get("rtlsim_vs_est")
            verdict = {"passed": "✅ pass", "failed": "❌ fail", "skipped": "skip", "running": "running"}.get(res, res)
            verdict = notes.get(ja["id"], verdict)
            print("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                model or mdir, board, size,
                "%.3f" % ratio if ratio is not None and res in ("passed", "failed") else "",
                "%.2f kB" % (m["fifo_kbit"] / 8) if m.get("fifo_kbit") is not None else "",
                fmt_time(t) if t is not None else "", fmt_time(tr) if tr is not None else "", verdict))


if __name__ == "__main__":
    main()
