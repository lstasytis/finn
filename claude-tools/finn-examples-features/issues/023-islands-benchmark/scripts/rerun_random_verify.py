#!/usr/bin/env python3
"""Re-run the random-frame verification (island build vs stitched-IP RTL) of a finished island
job whose reference simulation failed for a verify_accel reason, reusing the island netlist
simulation when it completed. Updates report/feature_metrics.json (verify.random, and
verify.random.rerun = true). Run inside the islands2 environment (wtenv.sh):

  python3 rerun_random_verify.py <job output dir> [variants, default: ref]
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.environ["FINN_ROOT"], "tests", "benchmark"))


def main():
    out = os.path.abspath(sys.argv[1])
    variants = sys.argv[2] if len(sys.argv) > 2 else "ref"
    mf = os.path.join(out, "report", "feature_metrics.json")
    m = json.load(open(mf))
    frames = m["verify"]["random"]["frames"]
    perf = json.load(open(os.path.join(m["prep_dir"], "report", "estimate_network_performance.json")))
    cycles = perf.get("critical_path_cycles", 0) + frames * perf.get("max_cycles", 0)
    vdir = os.path.join(out, "verify_random")
    script = os.path.join(os.environ["FINN_ROOT"], "experiments", "dynarapid", "verify_accel.py")
    cmd = [sys.executable, script, "--accel-dir", os.path.join(out, "islands"), "--out", vdir,
           "--frames", str(frames), "--vary-amplitude", "--lite-words", "0",
           "--max-polls", str(int(cycles / 2) + 50000), "--variants", variants]
    with open(os.path.join(vdir, "verify_rerun.log"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=vdir)
    s = json.load(open(os.path.join(vdir, "verify_accel.json")))
    keys = ("dr_done", "ref_done", "outputs_match", "ref_wrote_output", "out_head_dr", "out_head_ref",
            "output_bytes", "ref_written_bytes", "dr_complete", "outputs_match_where_ref_written")
    m["verify"]["random"].update({k: s.get(k) for k in keys}, rc=rc, rerun=True)
    with open(mf, "w") as f:
        json.dump(m, f, indent=2, default=str)
    print(json.dumps(m["verify"]["random"]))


if __name__ == "__main__":
    main()
