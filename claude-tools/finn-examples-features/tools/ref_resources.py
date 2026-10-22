"""Reference resources for the summary's res./ref. with the current resource estimate model.

    source tools/wtenv.sh build/wt/folding2 && python ref_resources.py [--force]

res./ref. divides a folding build's estimated resources by those of the committed folding built
the same way: the model's FIFO test build (committed folding + DWCs; FIFOs are excluded by the
reader). The FIFO builds ran on test2/fifo, whose estimates predate issue 021's estimate model, so
their own report/estimate_layer_resources.json is not comparable to the new folding/DWC results.
This re-estimates every FIFO build's step_generate_estimate_reports.onnx with the estimate code of
the worktree in $FINN_ROOT (the folding/DWC tests' code) and stores the per-layer estimate as
$REF_DIR/<fifo job id>.json (source build, estimate commit, layers). Entries whose source build
and commit are unchanged are skipped (issue 023).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_campaign import M, build_queue  # noqa: E402
from status_table import job_metrics, split_param  # noqa: E402

REF_DIR = os.path.join(M, "build/finn_build/fx/ref_resources")


def main():
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.analysis.fpgadataflow.res_estimation import res_estimation
    from finn.util.basic import part_map

    force = "--force" in sys.argv
    commit = subprocess.run(["git", "-C", os.environ["FINN_ROOT"], "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    os.makedirs(REF_DIR, exist_ok=True)
    for job in build_queue("2", ["small", "full"]):
        if job["feature"] != "fifo":
            continue
        m = job_metrics(job)
        src = os.path.join(m["output_dir"], "intermediate_models",
                           "step_generate_estimate_reports.onnx") if m else None
        if not src or not os.path.isfile(src):
            continue
        out = os.path.join(REF_DIR, job["id"] + ".json")
        try:
            with open(out) as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = {}
        if not force and old.get("source") == m["output_dir"] and old.get("commit") == commit:
            continue
        board, _ = split_param(job["param"])
        layers = ModelWrapper(src).analysis(lambda model: res_estimation(model, part_map[board]))
        with open(out, "w") as f:
            json.dump(dict(source=m["output_dir"], commit=commit, root=os.environ["FINN_ROOT"],
                           layers=layers), f, indent=1)
        print(job["id"], len(layers), "layers", flush=True)


if __name__ == "__main__":
    main()
