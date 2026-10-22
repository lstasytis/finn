"""Issue 023: move stale reference estimates aside and requeue the affected campaign jobs, each one
once its process has ended (running jobs are never killed).

    python requeue_watch.py [--dry-run]

* Stale reference estimates: every est_* dir in fx/folding2 and fx/dwc2 written before issue 021's
  estimate model reached those worktrees (or without a report), except the fresh ones built by
  regen_reference_estimate.py (suffix 00000021). They move to superseded_021/ next to them
  (the two resnet50 small ones named in issue 020 to superseded_020/), each as soon as no running
  campaign job of the same worktree uses its prefix.
* Requeued (requeue.sh): the finished and the running v2 fold-max / fold-target / dwc jobs (021
  estimates and budget, LabelSelect PE) except transformer_radioml/vision/language, and the v2 fifo /
  fifo-rtlsim jobs of the models whose depths change with 018. A job is requeued once its process
  has ended and its est prefix has no stale dir left (or has a fresh one). The three dwc jobs that
  issue 022's watcher requeues are left to it (only their estimates are moved).
"""
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

TOOLS = "/home/lstasytis/finn/claude-tools/finn-examples-features/tools"
sys.path.insert(0, TOOLS)
from run_campaign import LOGDIR, M, build_queue, pid_alive  # noqa: E402
from status_table import est_prefix  # noqa: E402

FX = os.path.join(M, "build/finn_build/fx")
EST_MODEL_SINCE = datetime.datetime(
    2026, 10, 22, 4, 29, 13, tzinfo=datetime.timezone.utc).timestamp()
FRESH = "00000021"
SUPERSEDED_020 = {"folding2/est_resnet50_small_V80_lclp_otw", "dwc2/est_resnet50_small_V80_3szgb2zd"}
EXCLUDE = re.compile(r"transformer_(radioml|vision|language)")
FOLD = re.compile(r"^v2__(fold-max|fold-target|dwc)__")
FIFO = re.compile(r"^v2__(fifo|fifo-rtlsim)__(mobilenet_v1|vgg10-radioml|resnet18|resnet50|espcn|bert)__")
LEFT_TO_022 = {"v2__dwc__yolov8__U55C-quantyolov8_4w4a__small",
               "v2__dwc__yolov8__U55C-quantyolov8_8w8a__small", "v2__dwc__resnet50__V80__small"}
LOG = os.path.join(LOGDIR, "requeue_023_watch.log")
WATCH_FILE = os.path.join(LOGDIR, "requeue_023_watch.json")
DRY = "--dry-run" in sys.argv


def log(msg):
    line = f"{datetime.datetime.now():%F %T} {msg}"
    print(line, flush=True)
    if not DRY:
        with open(LOG, "a") as f:
            f.write(line + "\n")


JOBS = {j["id"]: j for j in build_queue("2", ["small", "full"])}


def prefix_of(job):
    return job["wt"], est_prefix(job["model"], job["param"], job.get("size") == "small")


PREFIXES = {prefix_of(j) for j in JOBS.values() if j["wt"] in ("folding2", "dwc2")}


def dir_prefix(wt, name):
    cands = [p for w, p in PREFIXES if w == wt and name.startswith(p)]
    return max(cands, key=len) if cands else None


def stale_dirs():
    out = {}
    for wt in ("folding2", "dwc2"):
        for d in glob.glob(os.path.join(FX, wt, "est_*")):
            if not os.path.isdir(d) or d.endswith(FRESH):
                continue
            rep = os.path.join(d, "report", "estimate_layer_resources.json")
            if os.path.isfile(rep) and os.path.getmtime(rep) >= EST_MODEL_SINCE:
                continue
            # no report: stale only if old (a job may be building it right now)
            if not os.path.isfile(rep) and os.path.getmtime(d) >= EST_MODEL_SINCE:
                continue
            out[d] = (wt, dir_prefix(wt, os.path.basename(d)))
    return out


def has_fresh(wt, prefix):
    return os.path.isfile(os.path.join(FX, wt, prefix + FRESH, "report",
                                       "estimate_network_performance.json"))


def running():
    with open(os.path.join(LOGDIR, "campaign_state.json")) as f:
        state = json.load(f)
    return {j: v["pid"] for j, v in state.get("running", {}).items() if pid_alive(v["pid"])}


def main():
    run = running()
    if os.path.isfile(WATCH_FILE):
        # restart: the jobs still to requeue (the others already run again)
        with open(WATCH_FILE) as f:
            watch = json.load(f)
    else:
        affected = [j for j in JOBS if (FOLD.match(j) and not EXCLUDE.search(j)) or FIFO.match(j)]
        # finished (junit XML) or running now
        watch = {j: run.get(j) for j in affected
                 if j in run or os.path.isfile(os.path.join(LOGDIR, j + ".xml"))}
    log(f"start: {len(watch)} jobs ({sum(p is not None for p in watch.values())} running), "
        f"{len(stale_dirs())} stale est dirs")
    dead_since = {}
    while True:
        run = running()
        users = {prefix_of(JOBS[j]) for j in run if j in JOBS}
        stale = stale_dirs()
        for d, (wt, prefix) in sorted(stale.items()):
            if (wt, prefix) in users:
                continue
            rel = os.path.relpath(d, FX)
            dest = os.path.join(FX, wt, "superseded_020" if rel in SUPERSEDED_020 else "superseded_021")
            log(f"move {rel} -> {os.path.relpath(dest, FX)}/")
            if not DRY:
                os.makedirs(dest, exist_ok=True)
                shutil.move(d, dest)
            del stale[d]
        stale_prefixes = {v for v in stale.values()}
        batch = []
        for j, pid in list(watch.items()):
            if pid is not None and (j in run or pid_alive(pid)):
                continue
            dead_since.setdefault(j, time.time())
            if j in LEFT_TO_022:
                del watch[j]
                continue
            key = prefix_of(JOBS[j])
            if JOBS[j]["wt"] in ("folding2", "dwc2") and key in stale_prefixes and not has_fresh(*key):
                continue
            # the runner reaps a job within 20 s; give it 40
            if pid is not None and time.time() - dead_since[j] < 40:
                continue
            batch.append(j)
        if batch:
            log("requeue " + " ".join(batch))
            if not DRY:
                r = subprocess.run([os.path.join(TOOLS, "requeue.sh")] + batch,
                                   capture_output=True, text=True)
                log(f"requeue.sh exit {r.returncode}: {(r.stdout + r.stderr).strip()[-300:]}")
            for j in batch:
                del watch[j]
        if not DRY:
            with open(WATCH_FILE, "w") as f:
                json.dump(watch, f, indent=1)
        if DRY:
            log(f"dry run: waiting for {sorted(watch)}")
            return
        if not watch and not stale_dirs():
            log("all requeued, no stale est dirs left")
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
