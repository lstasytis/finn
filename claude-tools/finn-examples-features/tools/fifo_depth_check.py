"""Re-size the FIFOs of finished campaign FIFO builds with the code of another worktree and compare the depths
(tooling only): a build whose depths come out identical keeps its stitched-IP rtlsim result, the others need a
rerun.

    source tools/wtenv.sh <worktree> && python fifo_depth_check.py <worktree> [job-id regex] > depths.md
    python fifo_depth_check.py <worktree> --one <job id> <out json>      (internal: one job, in a subprocess)

Input: the build's intermediate_models/step_transpose_decomposition.onnx (the step before step_set_fifo_depths; the later saves of the phase already hold the FIFOs), sized with
the build's own configure_build from <worktree>; reference: its step_set_fifo_depths.onnx.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_campaign import build_queue  # noqa: E402
from status_table import MULTI, job_metrics, split_param  # noqa: E402

STRATEGY = {"fifo": "heuristic_analytical", "fifo-rtlsim": "heuristic_rtlsim"}


def fifo_depths(model):
    """{FIFO name: (depth, stream width in bits)} of a sized model."""
    from qonnx.custom_op.registry import getCustomOp

    out = {}
    for n in model.graph.node:
        if n.op_type.startswith("StreamingFIFO"):
            op = getCustomOp(n)
            out[n.name] = (op.get_nodeattr("depth"), op.get_instream_width())
    return out


def one(wt, jid, out_json):
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.builder.build_dataflow_steps import step_set_fifo_depths
    from finn.util.basic import make_build_dir

    job = next(j for j in build_queue("2", ["small", "full"]) if j["id"] == jid)
    mdir, (board, model) = job["model"], split_param(job["param"])
    bench = os.path.join(wt, "tests/benchmark")
    sys.path[:0] = [bench, os.path.join(bench, mdir)]
    spec = importlib.util.spec_from_file_location("tb", os.path.join(bench, mdir, f"test_build_{mdir}.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    src = job_metrics(job)["output_dir"]
    out = make_build_dir(f"depthcheck_{jid}_")
    args = (board, model, out) if mdir in MULTI else (board, out)
    cfg = tb.configure_build(*args, small=job.get("size") == "small", auto_fifo_depths=True,
                             auto_fifo_strategy=STRATEGY[job["feature"]], verify_steps=[])
    new = fifo_depths(step_set_fifo_depths(
        ModelWrapper(os.path.join(src, "intermediate_models", "step_transpose_decomposition.onnx")), cfg))
    ref = fifo_depths(ModelWrapper(os.path.join(src, "intermediate_models", "step_set_fifo_depths.onnx")))
    with open(out_json, "w") as f:
        json.dump(dict(src=src, new=new, ref=ref), f)


def main():
    wt = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--one":
        return one(wt, sys.argv[3], sys.argv[4])
    pat = re.compile(sys.argv[2] if len(sys.argv) > 2 else r"^v2__fifo__")
    outdir = os.path.join(os.environ["FINN_BUILD_DIR"], "depthcheck")
    os.makedirs(outdir, exist_ok=True)
    print("| job | FIFOs | changed | storage before | storage after | largest changes |")
    print("|---|---:|---:|---:|---:|---|")
    for job in build_queue("2", ["small", "full"]):
        if job["feature"] not in STRATEGY or not pat.search(job["id"]) or not job_metrics(job):
            continue
        res = os.path.join(outdir, job["id"] + ".json")
        if not os.path.exists(res):
            with open(os.path.join(outdir, job["id"] + ".log"), "w") as log:
                subprocess.run([sys.executable, os.path.abspath(__file__), wt, "--one", job["id"], res],
                               stdout=log, stderr=subprocess.STDOUT)
        if not os.path.exists(res):
            print(f"| {job['id']} | | | | | error, see {outdir}/{job['id']}.log |")
            continue
        with open(res) as f:
            d = json.load(f)
        new, ref = d["new"], d["ref"]
        changed = sorted((k for k in set(new) | set(ref) if new.get(k) != ref.get(k)),
                         key=lambda k: -abs((new.get(k) or [0])[0] - (ref.get(k) or [0])[0]))
        kb = lambda dd: sum(v[0] * v[1] for v in dd.values()) / 8192  # noqa: E731
        print("| %s | %d | %d | %.2f kB | %.2f kB | %s |" % (
            job["id"], len(ref), len(changed), kb(ref), kb(new),
            ", ".join(f"{k} {(ref.get(k) or ['-'])[0]}->{(new.get(k) or ['-'])[0]}" for k in changed[:4])))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
