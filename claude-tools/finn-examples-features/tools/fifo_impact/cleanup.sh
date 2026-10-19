#!/bin/bash
# Delete build scratch not needed any more: cppsim/rtlsim code gen, stitch projects, empty
# tav dirs, ipgen dirs the IP cache does not reference, and sol1/.autopilot of those it does.
B=/tmp/finn_dev_lstasytis/agents/fifo_clarity/build
python3 - <<'PY'
import json, os, shutil, time
B = "/tmp/finn_dev_lstasytis/agents/fifo_clarity/build"
idx = {}
try:
    idx = json.load(open(os.path.join(B, "featureflow_ipcache.json")))
except Exception:
    pass
keep = set()
for v in idx.values():
    for k in ("code_gen_dir_ipgen", "ipgen_path", "ip_path"):
        p = v.get(k, "")
        if p.startswith(B + "/"):
            keep.add(p[len(B) + 1 :].split("/")[0])
freed = 0
for name in os.listdir(B):
    path = os.path.join(B, name)
    if not os.path.isdir(path):
        continue
    if time.time() - os.path.getmtime(path) < 3 * 3600:
        continue  # may belong to a running build
    if name.startswith(("code_gen_cppsim_", "rtlsim_", "vivado_stitch_proj_", "tav_", "code_gen_ipgen_")) and name not in keep:
        shutil.rmtree(path, ignore_errors=True)
        freed += 1
    elif name in keep:
        for root, dirs, files in os.walk(path):
            if os.path.basename(root) == "sol1" and ".autopilot" in dirs:
                shutil.rmtree(os.path.join(root, ".autopilot"), ignore_errors=True)
print("removed", freed, "dirs; kept", len(keep), "ipgen dirs")
PY
du -sh $B | cut -f1; df -h /tmp/finn_dev_lstasytis | tail -1
