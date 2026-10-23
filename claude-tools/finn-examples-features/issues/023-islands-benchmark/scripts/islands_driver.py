#!/usr/bin/env python3
"""Gated driver of the island-flow benchmark jobs (issue 023) next to the running campaign.

Runs the pytest jobs of jobs.json (test2/islands worktree) in order, each pinned (taskset) to a
lane of CPUs that no other job of this driver uses. A job starts only while
  1-min load + its threads <= MAX_LOAD (two polls in a row), MemAvailable > MIN_MEM_GB,
  free disk >= MIN_DISK_GB, and START_GAP_S since the last start (the load average lags).
It never kills anything: when the room is gone it just stops launching.
A job in its verification phase (phase file says "verify": single-threaded xsim, affinity
widened by the test) no longer holds its lane.

  setsid nohup python3 islands_driver.py <state dir> &      (writes <state dir>/driver.pid)

jobs.json: [{"id", "node", "threads", "min_threads", "same_threads_as", "frames", "env": {},
             "after": [[id, "done"|"pass"], ...]}, ...]
state.json: per job: state (pending/running/pass/fail/skipped), pid, lane, start, end, rc, log.
"""

import json
import os
import shutil
import subprocess
import sys
import time

WT = "/home/lstasytis/finn/build/wt/islands2"
BUILD_DIR = "/home/lstasytis/finn/build/finn_build/fx/islands2"
ENV_SH = "/home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh"
MAX_LOAD = float(os.environ.get("MAX_LOAD", "115"))
MIN_MEM_GB = float(os.environ.get("MIN_MEM_GB", "200"))
MIN_DISK_GB = float(os.environ.get("MIN_DISK_GB", "60"))
START_GAP_S = float(os.environ.get("START_GAP_S", "180"))
# at most this many threads held by jobs still building (lanes of long, mostly serial prep
# phases add up while the 1-min load does not show them yet)
MAX_HELD = int(os.environ.get("MAX_HELD", "88"))
POLL_S = 30
# lanes of 4 physical cores + their SMT siblings (half a CCD of the EPYC 9554P), top first
LANE = 8
CCDS = [list(range(4 * c, 4 * c + 4)) + list(range(64 + 4 * c, 64 + 4 * c + 4)) for c in range(16)]
LANE_ORDER = list(range(15, -1, -1))


def mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return 0


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 2**30


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a zombie child counts as gone
    try:
        with open("/proc/%d/stat" % pid) as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def main():
    sdir = os.path.abspath(sys.argv[1])
    os.makedirs(os.path.join(sdir, "logs"), exist_ok=True)
    with open(os.path.join(sdir, "driver.pid"), "w") as f:
        f.write(str(os.getpid()))
    state_file = os.path.join(sdir, "state.json")
    state = json.load(open(state_file)) if os.path.isfile(state_file) else {}
    procs = {}
    last_start = 0.0
    ok_polls = 0
    last_ok = None

    def log(msg):
        with open(os.path.join(sdir, "driver.log"), "a") as f:
            f.write("%s %s\n" % (time.strftime("%F %T"), msg))

    def save():
        tmp = state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, state_file)

    log("driver start pid %d" % os.getpid())
    while True:
        jobs = json.load(open(os.path.join(sdir, "jobs.json")))
        if os.path.isfile(os.path.join(sdir, "STOP")):
            log("STOP file: no more launches")
        # reap
        for jid, st in state.items():
            if st["state"] == "running":
                p = procs.get(jid)
                done = (p.poll() is not None) if p is not None else not alive(st["pid"])
                if done:
                    rc = p.returncode if p is not None else None
                    if rc is None:
                        rc = _junit_rc(st.get("xml"))
                    st.update(state="pass" if rc == 0 else "fail", rc=rc, end=time.time())
                    log("%s finished rc=%s (%.0f s)" % (jid, rc, st["end"] - st["start"]))
                    procs.pop(jid, None)
                    save()
        # lanes held by jobs still building
        held = set()
        held_threads = 0
        for jid, st in state.items():
            if st["state"] == "running":
                phase = _phase(st)
                if phase != "verify":
                    held.update(st["lane"])
                    # (the prep phase is mostly serial HLS: counted as 2 threads)
                    held_threads += 2 if phase == "prep" else st["cpus"]
        free_lanes = [c for c in LANE_ORDER if c not in held]
        load = os.getloadavg()[0]
        eligible = []
        if not os.path.isfile(os.path.join(sdir, "STOP")):
            for j in jobs:
                st = state.get(j["id"], {"state": "pending"})
                if st["state"] != "pending":
                    continue
                deps = j.get("after", [])
                dep_states = [state.get(d, {}).get("state", "pending") for d, _ in deps]
                if any(
                    s in ("fail", "skipped") and cond == "pass"
                    for s, (_, cond) in zip(dep_states, deps)
                ):
                    state[j["id"]] = {"state": "skipped", "reason": "dependency failed"}
                    log("%s skipped (dependency failed)" % j["id"])
                    save()
                    continue
                if all(s in ("pass", "fail") for s in dep_states) and all(
                    s == "pass" or cond == "done" for s, (_, cond) in zip(dep_states, deps)
                ):
                    eligible.append(j)
        # the first eligible job (in list order) that fits now; a later one may go first when
        # an earlier one needs more threads than the load leaves
        nxt, threads, n_lanes = None, 0, 0
        common = (
            mem_avail_gb() > MIN_MEM_GB
            and disk_free_gb(BUILD_DIR) >= MIN_DISK_GB
            and time.time() - last_start >= START_GAP_S
        )
        for j in eligible if common else []:
            # thread cap: as many lanes as the load leaves room for, between min_threads and
            # threads; a baseline gets exactly the cap its island job had
            if j.get("same_threads_as"):
                want = lo = state[j["same_threads_as"]]["cpus"]
            else:
                want, lo = j["threads"], j.get("min_threads", j["threads"])
            room = int((MAX_LOAD - load) // LANE) * LANE
            room = min(room, (MAX_HELD - held_threads) // LANE * LANE)
            t = max(lo, min(want, room))
            if (
                load + t <= MAX_LOAD
                and len(free_lanes) >= max(1, t // LANE)
                and held_threads + t <= MAX_HELD
            ):
                nxt, threads, n_lanes = j, t, max(1, t // LANE)
                break
        # two polls in a row with room for the same job
        ok_polls = ok_polls + 1 if nxt is not None and nxt["id"] == last_ok else (1 if nxt else 0)
        last_ok = nxt["id"] if nxt else None
        if nxt is not None:
            if ok_polls >= 2:
                lanes = free_lanes[:n_lanes]
                cpus = sorted(c for ln in lanes for c in CCDS[ln])
                jid = nxt["id"]
                logf = os.path.join(sdir, "logs", jid + ".log")
                xml = os.path.join(sdir, "logs", jid + ".xml")
                phase_file = os.path.join(sdir, "logs", jid + ".phase")
                cmd = (
                    "cd %s && source %s . >/dev/null 2>&1 && "
                    "export %sISLANDS_VERIFY_FRAMES=%d ISLANDS_PHASE_FILE=%s "
                    "DYNARAPID_VIVADO_SLOTS=%d DYNARAPID_VIVADO_SLOTS_DIR=$FINN_BUILD_DIR/slots_lane%s "
                    "&& exec taskset -c %s python -m pytest -p no:cacheprovider -v -s "
                    "--junitxml=%s '%s'"
                    % (
                        WT,
                        ENV_SH,
                        "".join("%s=%s " % kv for kv in nxt.get("env", {}).items()),
                        nxt.get("frames", 1),
                        phase_file,
                        len(cpus),
                        "_".join(map(str, lanes)),
                        ",".join(map(str, cpus)),
                        xml,
                        nxt["node"],
                    )
                )
                p = subprocess.Popen(
                    ["bash", "-c", cmd],
                    stdout=open(logf, "w"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                procs[jid] = p
                last_start = time.time()
                ok_polls = 0
                state[jid] = dict(
                    state="running",
                    pid=p.pid,
                    lane=lanes,
                    cpus=len(cpus),
                    start=last_start,
                    load_start=load,
                    log=logf,
                    xml=xml,
                    phase_file=phase_file,
                    node=nxt["node"],
                )
                log("start %s pid %d lanes %s load %.1f mem %.0f GB" % (jid, p.pid, lanes, load, mem_avail_gb()))
                save()
        running = [j for j, s in state.items() if s["state"] == "running"]
        pending = [j for j in jobs if state.get(j["id"], {"state": "pending"})["state"] == "pending"]
        if not running and (not pending or os.path.isfile(os.path.join(sdir, "STOP"))):
            log("nothing left to run, driver exits")
            break
        time.sleep(POLL_S)


def _phase(st):
    try:
        with open(st["phase_file"]) as f:
            return f.read().strip()
    except OSError:
        return "build"


def _junit_rc(xml):
    try:
        txt = open(xml).read()
    except (OSError, TypeError):
        return -1
    return 0 if ('failures="0"' in txt and 'errors="0"' in txt) else 1


if __name__ == "__main__":
    main()
