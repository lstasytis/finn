"""Feature-test campaign scheduler (tooling, not part of any PR).

    python run_campaign.py [--plan] [--only REGEX] [--deadline HH:MM|YYYY-MM-DDTHH:MM|none] [--max-threads N]

Runs every (test branch worktree, pytest node id) job of the queue below as its own pytest
process, in queue order, while
  * our reserved threads (sum of each running job's NUM_DEFAULT_WORKERS + 1) stay within the
    budget (default: 102 minus the other users' load, sampled at start) and the 1-min load
    average stays below 102,
  * MemAvailable minus the job's memory estimate stays above 151 GB (20 %).
Watchdog: below 151 GB free no job is launched; below 75 GB the newest job's process tree is
killed (and requeued once). At the deadline every running job is killed (result "aborted").
Restartable: jobs whose junit XML says passed/skipped are not rerun (failed ones are rerun only
with --rerun-failed). Per-job logs and junit XML: $LOGDIR/<job>.{log,xml}; state in
$LOGDIR/campaign_state.json; the status table is regenerated after every finished job.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

M = "/home/lstasytis/finn"
TOOLS = os.path.join(M, "claude-tools/finn-examples-features/tools")
LOGDIR = os.path.join(M, "build/finn_build/fx/logs/campaign")
GIB = 1024**3

# (model dir, test stem, [(param id, tier, workers, mem GiB)])
TIER1 = [
    ("bnn-pynq", "bnnpynq", [
        (f"AUP-ZU3_8GB-{m}", 1, 4, 8) for m in
        ["tfc-w1a1", "tfc-w1a2", "tfc-w2a2", "lfc-w1a1", "lfc-w1a2", "cnv-w1a1", "cnv-w1a2",
         "cnv-w2a2"]
    ] + [("KV260_SOM-cnv-w1a2", 1, 4, 8), ("U55C-cnv-w2a2", 1, 4, 8)]),
    ("gtsrb", "gtsrb", [("AUP-ZU3_8GB", 1, 4, 8)]),
    ("kws", "kws", [("AUP-ZU3_8GB", 1, 4, 8)]),
    ("cybersecurity-mlp", "cybersecuritymlp", [("AUP-ZU3_8GB", 1, 6, 12)]),
    ("vgg10-radioml", "vgg10radioml", [("ZCU104", 1, 6, 12)]),
    ("espcn", "espcn", [("KV260_SOM", 1, 6, 16)]),
]
TIER2 = [
    ("mobilenet_v1", "mobilenetv1", [("U55C", 2, 8, 32)]),
    ("resnet18", "resnet18", [("U55C-resnet18_4w4a", 2, 8, 48), ("U55C-resnet18_8w8a", 2, 8, 64)]),
    ("yolov8", "yolov8", [("U55C-quantyolov8_4w4a", 2, 8, 48), ("U55C-quantyolov8_8w8a", 2, 8, 64)]),
    ("bert", "bert", [("VCK190", 2, 8, 64)]),
    ("resnet50", "resnet50", [("V80", 2, 8, 64)]),
]
# variant 2 (new upstream base) only: the finn+ transformers of upstream #1725
TIER_TRANSFORMER = [
    ("transformer", "transformer", [
        ("ZCU104-transformer_benchmark", 1, 4, 8), ("ZCU104-transformer_radioml", 1, 4, 8),
        ("ZCU104-transformer_vision", 2, 6, 24), ("VCK190-transformer_language", 2, 8, 48),
    ]),
]

# (feature, worktree, test suffix template)
FEATURES = [
    ("fifo", "fifo", "fifo_sizing[{p}-heuristic_analytical]"),
    ("fold-target", "folding", "folding[{p}-target]"),
    ("fold-max", "folding", "folding[{p}-maximize]"),
    ("dwc", "dwc", "dwc[{p}]"),
    # optional FIFO-aware folding (slower): once per model, full size only
    ("fold-fifo", "folding", "folding[{p}-fifo-aware]"),
]
LAST = [("fifo-rtlsim", "fifo", "fifo_sizing[{p}-heuristic_rtlsim]")]


def build_queue(variant="", sizes=None):
    """The jobs of one variant: "" (the 2026-10-19 base, worktrees fifo/folding/dwc) or e.g.
    "2" (worktrees fifo2/..., job ids prefixed "v2__", plus the transformers). With ``sizes``
    (e.g. ["small", "full"]) every job exists per folding size (test id suffix "-<size>", job id
    suffix "__small" for the small one); a full job runs only after its small sibling passed or
    skipped ("after"), and all small jobs come first. Maximize has no small variant."""
    if sizes:
        base = build_queue(variant)
        small = [
            dict(j, id=j["id"] + "__small", node=j["node"][:-1] + "-small]", size="small")
            for j in base
            if "small" in sizes and j["feature"] not in ("fold-max", "fold-fifo")
        ]
        full = [
            dict(j, node=j["node"][:-1] + "-full]", size="full",
                 after=j["id"] + "__small"
                 if j["feature"] not in ("fold-max", "fold-fifo") and "small" in sizes else None)
            for j in base
            if "full" in sizes
        ]
        return small + full
    jobs = []
    tier2 = TIER2 + (TIER_TRANSFORMER if variant else [])
    # small models first, but the large models' FIFO tests early: they need the most time;
    # heuristic_rtlsim FIFO sizing last
    order = [(FEATURES[:1], TIER1), (FEATURES[:1], tier2), (FEATURES[1:], TIER1),
             (FEATURES[1:], tier2), (LAST, TIER1), (LAST, tier2)]
    prefix = f"v{variant}__" if variant else ""
    for features, tier in order:
        for feat, wt, tmpl in features:
            for mdir, stem, params in tier:
                for p, t, workers, mem in params:
                    name, func = tmpl.split("[", 1)
                    node = f"tests/benchmark/{mdir}/test_build_{mdir}.py::test_{stem}_{name}[{func}"
                    node = node.format(p=p)
                    jobs.append(dict(
                        id=f"{prefix}{feat}__{mdir}__{p}", feature=feat, wt=wt + variant,
                        node=node, model=mdir, param=p, tier=t, workers=workers, mem=mem,
                    ))
    return jobs


def junit_result(xml):
    if not os.path.isfile(xml):
        return None
    try:
        case = ET.parse(xml).getroot().iter("testcase").__next__()
    except Exception:
        return "error"
    for tag in ("failure", "error"):
        if case.find(tag) is not None:
            return "failed"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def meminfo(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


def mem_available():
    return meminfo("MemAvailable")


def mem_total():
    return meminfo("MemTotal")


def kill_tree(pid, sig=signal.SIGKILL):
    try:
        children = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    except Exception:
        children = ""
    for c in children.split():
        kill_tree(int(c), sig)
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    with open(f"/proc/{pid}/stat") as f:
        return f.read().split(") ")[1][0] != "Z"


class Adopted:
    """A job process started by an earlier runner (not our child)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            return None if pid_alive(self.pid) else 0
        except OSError:
            return 0


def launch(job):
    wt = os.path.join(M, "build/wt", job["wt"])
    log, xml = (os.path.join(LOGDIR, job["id"] + ext) for ext in (".log", ".xml"))
    cmd = (
        f"source {TOOLS}/wtenv.sh {wt} && cd {wt} && export NUM_DEFAULT_WORKERS={job['workers']} "
        f"&& exec python -m pytest -p no:cacheprovider -v '{job['node']}' --junitxml={xml}"
    )
    with open(log, "w") as f:
        f.write(f"# {datetime.datetime.now()} {cmd}\n")
        f.flush()
        proc = subprocess.Popen(
            ["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only", default=None, help="regex on job ids")
    ap.add_argument("--deadline", default="19:15")
    ap.add_argument("--max-threads", type=int, default=None)
    ap.add_argument("--max-load", type=float, default=102, help="no launch above this 1-min load")
    ap.add_argument("--rerun-failed", action="store_true")
    ap.add_argument("--variant", default="", help="queue this variant's jobs (e.g. 2)")
    ap.add_argument("--first", action="append", default=[], help="regex: run these first")
    ap.add_argument("--sizes", default=None, help="e.g. small,full: per-size jobs (see build_queue)")
    args = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)
    pid_file = os.path.join(LOGDIR, "runner.pid")
    if os.path.isfile(pid_file) and not args.plan:
        with open(pid_file) as f:
            other = int(f.read().strip() or 0)
        if other and other != os.getpid() and pid_alive(other):
            sys.exit(f"another runner is alive (pid {other}); stop it first")
    if not args.plan:
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
    sizes = args.sizes.split(",") if args.sizes else None
    jobs = build_queue(args.variant, sizes)
    # running jobs of any variant are adopted, so that their threads and memory count
    known = {j["id"]: j for v in dict.fromkeys(["", args.variant]) for j in build_queue(v)}
    known.update({j["id"]: j for j in jobs})
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["id"])]
    # priority groups in the order given, the rest after them (stable within a group)
    jobs.sort(key=lambda j: next(
        (i for i, rx in enumerate(args.first) if re.search(rx, j["id"])), len(args.first)))
    if args.plan:
        for j in jobs:
            print(j["id"], j["node"], junit_result(os.path.join(LOGDIR, j["id"] + ".xml")))
        return
    if args.deadline == "none":
        deadline = datetime.datetime.max
    elif "T" in args.deadline:
        deadline = datetime.datetime.fromisoformat(args.deadline)
    else:
        hh, mm = map(int, args.deadline.split(":"))
        deadline = datetime.datetime.now().replace(hour=hh, minute=mm, second=0)
    others = os.getloadavg()[0]
    budget = args.max_threads or int(102 - others)
    print(f"budget {budget} threads (others' load {others:.0f}), deadline {deadline}", flush=True)
    state_file = os.path.join(LOGDIR, "campaign_state.json")
    pending = []
    for j in jobs:
        res = junit_result(os.path.join(LOGDIR, j["id"] + ".xml"))
        if res in ("passed", "skipped") or (res == "failed" and not args.rerun_failed):
            continue
        pending.append(j)
    running = {}  # id -> (job, proc, start)
    requeued = set()
    old = {}
    if os.path.isfile(state_file):
        with open(state_file) as f:
            old = json.load(f)
    status = old.get("finished", {})
    for jid, info in old.get("running", {}).items():
        job = next((j for j in pending if j["id"] == jid), None) or known.get(jid)
        if job is not None and pid_alive(info["pid"]):
            running[jid] = (job, Adopted(info["pid"]), info["start"])
            if job in pending:
                pending.remove(job)
            print(f"adopted {jid} (pid {info['pid']})", flush=True)

    def save():
        with open(state_file, "w") as f:
            json.dump(dict(
                updated=str(datetime.datetime.now()), budget=budget,
                running={k: dict(start=v[2], pid=v[1].pid, node=v[0]["node"]) for k, v in running.items()},
                pending=[j["id"] for j in pending], finished=status,
            ), f, indent=1)

    def regen_status():
        extra = (["--variant", args.variant] if args.variant else []) + (
            ["--sizes", args.sizes] if args.sizes else [])
        subprocess.run([sys.executable, os.path.join(TOOLS, "status_table.py")] + extra,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    last_regen = time.time()
    while pending or running:
        now = datetime.datetime.now()
        if time.time() - last_regen > 600:  # running/pending markers, also without completions
            regen_status()
            last_regen = time.time()
        # reap
        for jid, (job, proc, start) in list(running.items()):
            if proc.poll() is not None:
                res = junit_result(os.path.join(LOGDIR, jid + ".xml")) or f"exit{proc.returncode}"
                status[jid] = dict(result=res, duration_s=time.time() - start)
                print(f"{now:%H:%M} done {jid}: {res} ({(time.time() - start) / 60:.0f} min)", flush=True)
                del running[jid]
                regen_status()
        if now >= deadline:
            for jid, (job, proc, start) in list(running.items()):
                kill_tree(proc.pid)
                status[jid] = dict(result="aborted", duration_s=time.time() - start)
                print(f"{now:%H:%M} aborted {jid} (deadline)", flush=True)
            running.clear()
            save()
            regen_status()
            break
        free = mem_available()
        # watchdog
        if free < 75 * GIB and running:
            jid = max(running, key=lambda k: running[k][2])
            job, proc, start = running.pop(jid)
            kill_tree(proc.pid)
            print(f"{now:%H:%M} WATCHDOG killed {jid} (MemAvailable {free / GIB:.0f} GiB)", flush=True)
            if jid not in requeued:
                requeued.add(jid)
                pending.insert(0, job)
            else:
                status[jid] = dict(result="killed-oom", duration_s=time.time() - start)
        # launch
        used = sum(j["workers"] + 1 for j, _, _ in running.values())
        for job in list(pending):
            dep = job.get("after")
            if dep:
                res = junit_result(os.path.join(LOGDIR, dep + ".xml"))
                queued = dep in running or any(j["id"] == dep for j in pending)
                if res is None and queued:
                    continue  # the small sibling has not finished yet
                if res not in (None, "passed", "skipped"):
                    pending.remove(job)
                    status[job["id"]] = dict(result="blocked", duration_s=0, by=dep)
                    print(f"{now:%H:%M} blocked {job['id']} ({dep}: {res})", flush=True)
                    continue
            need = job["workers"] + 1
            if used + need > budget or os.getloadavg()[0] > args.max_load:
                break
            if mem_available() - job["mem"] * GIB < 151 * GIB:
                break
            # reservations of the jobs of the last hour too: they have not grown yet (older
            # ones show in MemAvailable)
            young = sum(j["mem"] for j, _, t0 in running.values() if time.time() - t0 < 3600)
            if mem_available() - (young + job["mem"]) * GIB < 151 * GIB:
                break
            if shutil.disk_usage(M).free < 60 * GIB:
                break
            proc = launch(job)
            running[job["id"]] = (job, proc, time.time())
            pending.remove(job)
            used += need
            print(f"{now:%H:%M} start {job['id']} (pid {proc.pid}, {used}/{budget} threads)", flush=True)
            time.sleep(2)
        save()
        time.sleep(20)
    save()
    regen_status()
    print("campaign finished", flush=True)


if __name__ == "__main__":
    main()
