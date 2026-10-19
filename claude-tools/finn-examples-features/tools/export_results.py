"""Copy the campaign's results into the repo (claude-tools/finn-examples-features/results/<name>/).

Per job: junit XML, the pytest log tail, and the latest matching build's small reports
(feature_metrics.json, estimate/rtlsim reports, auto_folding_config.json, final_hw_config.json,
time_per_step.json); plus the reference estimate reports, the runner log and state.
    python export_results.py [name]
"""
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_campaign import LOGDIR, M, build_queue  # noqa: E402
from status_table import TAGS, build_prefix  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "campaign_20261019"
out = os.path.join(M, "claude-tools/finn-examples-features/results", name)
os.makedirs(out, exist_ok=True)
FILES = ["report/feature_metrics.json", "report/estimate_network_performance.json",
         "report/estimate_layer_cycles.json", "report/estimate_layer_resources.json",
         "report/rtlsim_performance.json", "auto_folding_config.json", "final_hw_config.json",
         "time_per_step.json"]
for job in build_queue():
    d = os.path.join(out, "jobs", job["id"])
    xml = os.path.join(LOGDIR, job["id"] + ".xml")
    log = os.path.join(LOGDIR, job["id"] + ".log")
    if not os.path.isfile(xml) and not os.path.isfile(log):
        continue
    os.makedirs(d, exist_ok=True)
    if os.path.isfile(xml):
        shutil.copy(xml, d)
    if os.path.isfile(log):
        with open(log, errors="replace") as f:
            lines = f.read().replace("\0", "").splitlines()
        with open(os.path.join(d, "pytest_tail.log"), "w") as f:
            f.write("\n".join(lines[-150:]) + "\n")
    cands = glob.glob(os.path.join(M, "build/finn_build/fx", job["wt"],
                                   build_prefix(TAGS[job["feature"]], job["model"], job["param"]) + "*"))
    cands = [c for c in cands if os.path.isdir(c)]
    if cands:
        b = max(cands, key=os.path.getmtime)
        with open(os.path.join(d, "build_dir.txt"), "w") as f:
            f.write(b + "\n")
        for rel in FILES + [os.path.relpath(p, b) for p in glob.glob(b + "/verification_output/*")]:
            src = os.path.join(b, rel)
            if os.path.isfile(src) and os.path.getsize(src) < 2_000_000:
                dst = os.path.join(d, rel.replace("/", "__"))
                shutil.copy(src, dst)
# reference estimates
for e in glob.glob(os.path.join(M, "build/finn_build/fx/*/est_*/report/estimate_network_performance.json")):
    parts = e.split("/")
    wt, est = parts[-4], parts[-3]
    if wt not in ("fifo", "folding", "dwc"):
        continue
    d = os.path.join(out, "reference_estimates", wt, est)
    os.makedirs(d, exist_ok=True)
    shutil.copy(e, d)
for f in ["runner.log", "campaign_state.json"]:
    if os.path.isfile(os.path.join(LOGDIR, f)):
        shutil.copy(os.path.join(LOGDIR, f), out)
print(out)
