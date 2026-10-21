"""Vanilla baseline (tooling only): upstream's own default flow for the models new in upstream PR #1725,
on pure upstream/update/finn-examples (worktree build/wt/vanilla), up to and including stitched-IP rtlsim
(verification + performance), no bitfile.

    python run_vanilla.py launch            # all builds in parallel, smallest expected build time first
    python run_vanilla.py one <mdir> <board> [<model>]
    python run_vanilla.py report            # results table

Each build uses the test file's own configure_build (committed folding, upstream's FIFO sizing, its verify steps),
only generate_outputs is reduced to estimate reports + stitched IP + rtlsim performance.
"""
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time

WT = "/home/lstasytis/finn/build/wt/vanilla"
TOOLS = os.path.dirname(os.path.abspath(__file__))
OUTROOT = "/home/lstasytis/finn/build/finn_build/fx/vanilla"
LOGS = os.path.join(OUTROOT, "logs")
# smallest expected build time first (campaign durations of the same models)
BUILDS = [
    ("transformer", "ZCU104", "transformer_benchmark", 4),
    ("transformer", "ZCU104", "transformer_radioml", 4),
    ("bert", "VCK190", None, 8),
    ("transformer", "ZCU104", "transformer_vision", 6),
    ("espcn", "KV260_SOM", None, 6),
    ("transformer", "VCK190", "transformer_language", 8),
    ("resnet18", "U55C", "resnet18_4w4a", 8),
    ("resnet18", "U55C", "resnet18_8w8a", 8),
    ("yolov8", "U55C", "quantyolov8_4w4a", 8),
    ("yolov8", "U55C", "quantyolov8_8w8a", 8),
]


def tag(mdir, board, model):
    return f"{model or mdir}_{board}"


def one(mdir, board, model=None):
    bench = os.path.join(WT, "tests/benchmark")
    sys.path[:0] = [bench, os.path.join(bench, mdir)]
    spec = importlib.util.spec_from_file_location("tb", os.path.join(bench, mdir, f"test_build_{mdir}.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    import finn.builder.build_dataflow as build
    import finn.builder.build_dataflow_config as build_cfg
    from finn.util.basic import make_build_dir

    out = make_build_dir(f"vanilla_{tag(mdir, board, model)}_")
    args = (board, model) if model else (board,)
    cfg = tb.configure_build(
        *args, out,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
                          build_cfg.DataflowOutputType.STITCHED_IP,
                          build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE],
    )
    cfg.enable_build_pdb_debug = False
    model_file = tb.get_model_file(model) if model else tb.model_file
    t0 = time.time()
    ret = build.build_dataflow_cfg(model_file, cfg)
    res = dict(model=model or mdir, board=board, output_dir=out, ret=ret, minutes=round((time.time() - t0) / 60, 1))
    res["verify"] = {os.path.basename(f): ("SUCCESS" if "_SUCCESS" in f else "FAIL")
                     for f in sorted(glob.glob(os.path.join(out, "verification_output", "verify_*.npy")))}
    for name, rel in (("estimate", "report/estimate_network_performance.json"),
                      ("rtlsim", "report/rtlsim_performance.json")):
        try:
            res[name] = json.load(open(os.path.join(out, rel)))
        except (OSError, ValueError):
            pass
    try:
        log = open(os.path.join(out, "build_dataflow.log")).read().splitlines()
        res["error"] = next((ln for ln in reversed(log) if "Error" in ln or "error" in ln), None)
    except OSError:
        pass
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, tag(mdir, board, model) + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("estimate", "rtlsim")}, indent=1))


def launch():
    os.makedirs(LOGS, exist_ok=True)
    env_sh = os.path.join(TOOLS, "wtenv.sh")
    for mdir, board, model, workers in BUILDS:
        t = tag(mdir, board, model)
        cmd = (f"source {env_sh} {WT} >/dev/null 2>&1 && export FINN_BUILD_DIR={OUTROOT} "
               f"NUM_DEFAULT_WORKERS={workers} && echo $$ > {LOGS}/{t}.pid && "
               f"exec python {os.path.abspath(__file__)} one {mdir} {board} {model or ''}")
        with open(os.path.join(LOGS, t + ".log"), "w") as log:
            subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        print("launched", t, flush=True)
        time.sleep(5)


def report():
    for mdir, board, model, _ in BUILDS:
        t = tag(mdir, board, model)
        p = os.path.join(LOGS, t + ".json")
        if not os.path.isfile(p):
            pid = open(os.path.join(LOGS, t + ".pid")).read().strip() if os.path.isfile(
                os.path.join(LOGS, t + ".pid")) else None
            alive = pid and os.path.exists(f"/proc/{pid}")
            print(f"{t}: {'running' if alive else 'not finished (no result)'}")
            continue
        r = json.load(open(p))
        est = (r.get("estimate") or {}).get("estimated_throughput_fps")
        rt = (r.get("rtlsim") or {}).get("stable_throughput[images/s]")
        ver = ", ".join(f"{k.split('verify_')[1].rsplit('_', 2)[0]}:{v}" for k, v in r["verify"].items())
        print(f"{t}: ret {r['ret']}, {r['minutes']} min, verify [{ver}], est {est}, rtlsim {rt}"
              + (f", error: {r['error'][:160]}" if r["ret"] != 0 and r.get("error") else ""))


if __name__ == "__main__":
    {"launch": lambda: launch(), "report": lambda: report(),
     "one": lambda: one(*sys.argv[2:])}[sys.argv[1]]()
