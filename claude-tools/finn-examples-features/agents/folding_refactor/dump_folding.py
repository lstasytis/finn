#!/usr/bin/env python3
"""Behaviour dump of SetFolding on the convert-phase checkpoints of the per-model feature tests.

Runs a given ``set_folding.py`` (loaded from a file, so the original and the refactored version
can be compared in the same FINN tree) on every finished reference estimate build
``$CKPT_ROOT/est_<model>_<board>_*`` in several modes and writes one JSON per (model, mode):
every node of the folded graph (all attributes, tensor shapes, initializer hashes, FINNLoop
bodies recursively), ``dataflow_performance(model)`` and SetFolding's ``last_search_report``.
Timings go to ``_timing.json`` only, so two dump directories compare with ``diff -r``.

Driver:  dump_folding.py --impl set_folding.py --out DIR [--jobs 14] [--models RE] [--modes a,b]
Worker:  dump_folding.py --worker --impl set_folding.py --ckpt EST_DIR --mode M --out FILE

Run it with a FINN environment on PYTHONPATH (wtenv.sh); the SetFolding call replicates
step_target_fps_parallelization with the cfg defaults and the feature tests' overrides.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

CKPT_ROOT = "/home/lstasytis/finn/build/finn_build/fx/folding"
CKPT = "intermediate_models/phase_convert_to_hardware.onnx"
REPORT = "report/estimate_network_performance.json"
MODES = ["naive", "target", "maximize", "padded", "fifo"]
# per-model cfg fields that differ from the DataflowBuildConfig defaults (tests/benchmark/*)
MODEL_CFG = {"cybersecurity-mlp": {"mvau_wwidth_max": 80}}


def builder_kwargs(model_name, board, target, mode):
    """SetFolding kwargs as step_target_fps_parallelization passes them (cfg defaults plus the
    feature tests' OPTIMIZER_FOLDING), with a fixed seed so that the optimizer is deterministic."""
    kw = dict(
        mvau_wwidth_max=1024,
        two_pass_relaxation=True,
        style="optimizer",
        folding_maximum_padding=0,
        enable_folding_dwc_heuristic=True,
        enable_folding_fifo_heuristic=False,
        folding_effort=50,
        folding_max_attempts=1,
        folding_pad_io_nodes=False,
        prefer_memory=None,
        prefer_compute=None,
        platform=board,
        auto_fifo_strategy="heuristic_analytical",
        strict_budget=False,
        allow_uram_weights=False,
        seed=0,
    )
    kw.update(MODEL_CFG.get(model_name, {}))
    if mode == "naive":
        kw["style"] = "naive"
    elif mode == "maximize":
        target = None
    elif mode == "padded":
        kw["folding_maximum_padding"] = 4
    elif mode == "fifo":
        kw["enable_folding_fifo_heuristic"] = True
    return target, kw


def find_checkpoints(root, model_re):
    from finn.util.basic import part_map

    boards = sorted(part_map, key=len, reverse=True)
    found = []
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if not d.startswith("est_") or not os.path.isfile(os.path.join(path, REPORT)):
            continue
        if not os.path.isfile(os.path.join(path, CKPT)):
            continue
        stem = d[len("est_") : -9]  # make_build_dir's random suffix: "_" + 8 characters
        board = next((b for b in boards if stem.endswith("_" + b)), None)
        if board is None:
            continue
        model = stem[: -len(board) - 1]
        if model_re and not re.search(model_re, model):
            continue
        found.append((model, board, path))
    return found


# ---------------------------------------------------------------- worker


def _tensor_hash(arr):
    import numpy as np

    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def _attr(a):
    from onnx import helper, numpy_helper

    v = helper.get_attribute_value(a)
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if hasattr(v, "node") and hasattr(v, "initializer"):  # GraphProto (FINNLoop body)
        return _graph(v, None)
    if hasattr(v, "raw_data") or hasattr(v, "dims"):  # TensorProto
        return _tensor_hash(numpy_helper.to_array(v))
    if isinstance(v, list):
        return [x.decode(errors="replace") if isinstance(x, bytes) else x for x in v]
    return v


def _graph(graph, model):
    from onnx import numpy_helper

    def vinfo(name):
        if model is None:
            return name
        return [name, model.get_tensor_shape(name), str(model.get_tensor_datatype(name))]

    out = {
        "inputs": [vinfo(x.name) for x in graph.input],
        "outputs": [vinfo(x.name) for x in graph.output],
        "nodes": [],
        "initializers": {
            t.name: _tensor_hash(numpy_helper.to_array(t)) for t in graph.initializer
        },
    }
    for n in graph.node:
        out["nodes"].append(
            {
                "name": n.name,
                "op_type": n.op_type,
                "domain": n.domain,
                "inputs": [vinfo(x) for x in n.input],
                "outputs": [vinfo(x) for x in n.output],
                "attrs": {a.name: _attr(a) for a in n.attribute},
            }
        )
    if model is not None:
        # datatype annotations of initializers (e.g. retyped padded weights)
        out["initializer_dtypes"] = {
            t.name: str(model.get_tensor_datatype(t.name)) for t in graph.initializer
        }
    return out


def _jsonable(x):
    import numpy as np

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def worker(args):
    import importlib.util
    import traceback
    import warnings

    from qonnx.core.modelwrapper import ModelWrapper

    from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance

    spec = importlib.util.spec_from_file_location("set_folding_under_test", args.impl)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt = args.ckpt.rstrip("/")
    model_name, board = args.model, args.board
    with open(os.path.join(ckpt, REPORT)) as f:
        ref_cycles = int(json.load(f)["max_cycles"])
    target, kw = builder_kwargs(model_name, board, ref_cycles, args.mode)
    dump = {
        "model": model_name,
        "board": board,
        "mode": args.mode,
        "checkpoint": os.path.basename(ckpt),
        "target_cycles_per_frame": target,
        "kwargs": kw,
    }
    t0 = time.time()
    try:
        model = ModelWrapper(os.path.join(ckpt, CKPT))
        folder = mod.SetFolding(target, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model.transform(folder, apply_to_subgraphs=True)
        dump["result"] = "ok"
        dump["performance"] = model.analysis(dataflow_performance)
        dump["search_report"] = folder.last_search_report
        dump["graph"] = _graph(model.graph, model)
    except Exception as e:  # noqa: BLE001 - an exception is behaviour too
        dump["result"] = "error"
        dump["error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else "")
        traceback.print_exc()
    elapsed = time.time() - t0
    with open(args.out, "w") as f:
        json.dump(_jsonable(dump), f, sort_keys=True, indent=1)
    with open(args.out + ".time", "w") as f:
        f.write("%.3f\n" % elapsed)


# ---------------------------------------------------------------- driver


def driver(args):
    os.makedirs(args.out, exist_ok=True)
    modes = args.modes.split(",") if args.modes else MODES
    if args.runs_from:
        # the same (model, board, checkpoint, mode) list as an earlier dump
        with open(os.path.join(args.runs_from, "_runs.json")) as f:
            runs = [tuple(r) for r in json.load(f)]
        runs = [r for r in runs if r[3] in modes and (not args.models or re.search(args.models, r[0]))]
    else:
        runs = [
            (m, b, p, mode)
            for (m, b, p) in find_checkpoints(args.ckpt_root, args.models)
            for mode in modes
        ]
    with open(os.path.join(args.out, "_runs.json"), "w") as f:
        json.dump(runs, f, indent=1)
    if args.exclude:
        runs = [r for r in runs if not re.search(args.exclude, "%s:%s" % (r[0], r[3]))]
    # longest first: maximize/padded/fifo of the big models
    weight = {"maximize": 4, "padded": 3, "fifo": 3, "target": 1, "naive": 0}
    runs.sort(key=lambda r: -weight[r[3]])
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = "1"
    env["NUM_DEFAULT_WORKERS"] = "1"
    env["PYTHONHASHSEED"] = "0"
    timing, pending, running = {}, list(runs), {}
    log = open(os.path.join(args.out, "_driver.log"), "a")
    print("%d runs, %d jobs" % (len(runs), args.jobs), file=log, flush=True)
    while pending or running:
        while pending and len(running) < args.jobs:
            m, b, p, mode = pending.pop(0)
            key = "%s__%s__%s" % (m, b, mode)
            out = os.path.join(args.out, key + ".json")
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--impl", args.impl]
            cmd += ["--ckpt", p, "--mode", mode, "--model", m, "--board", b, "--out", out]
            if args.cpus:
                cmd = ["taskset", "-c", args.cpus] + cmd
            errf = open(os.path.join(args.out, key + ".log"), "w")
            proc = subprocess.Popen(
                cmd, env=env, stdout=errf, stderr=subprocess.STDOUT, start_new_session=True
            )
            running[key] = (proc, time.time(), out, errf)
            print("start %s pid %d" % (key, proc.pid), file=log, flush=True)
        time.sleep(1)
        for key, (proc, t0, out, errf) in list(running.items()):
            if proc.poll() is None and time.time() - t0 <= args.timeout:
                continue
            if proc.poll() is None:
                os.killpg(proc.pid, 9)
                proc.wait()
                with open(out, "w") as f:
                    json.dump({"result": "timeout", "timeout_s": args.timeout}, f)
                timing[key] = "timeout"
            else:
                try:
                    with open(out + ".time") as f:
                        timing[key] = float(f.read())
                    os.remove(out + ".time")
                except OSError:
                    timing[key] = "rc=%d" % proc.returncode
                    with open(out, "w") as f:
                        json.dump({"result": "crash", "returncode": proc.returncode}, f)
            errf.close()
            del running[key]
            print("done %s %s" % (key, timing[key]), file=log, flush=True)
    tpath = os.path.join(args.out, "_timing.json")
    old = {}
    if os.path.isfile(tpath):
        with open(tpath) as f:
            old = json.load(f)
    old.update(timing)
    with open(tpath, "w") as f:
        json.dump(old, f, sort_keys=True, indent=1)
    print("finished", file=log, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, help="set_folding.py to load")
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--ckpt")
    ap.add_argument("--mode")
    ap.add_argument("--model")
    ap.add_argument("--board")
    ap.add_argument("--ckpt-root", default=CKPT_ROOT)
    ap.add_argument("--models", default=None, help="regex on the model name")
    ap.add_argument("--modes", default=None, help="comma-separated subset of " + ",".join(MODES))
    ap.add_argument("--runs-from", default=None, help="dump dir whose _runs.json to repeat")
    ap.add_argument("--exclude", default=None, help="regex on '<model>:<mode>' to skip")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--cpus", default=None, help="taskset CPU list for the workers")
    args = ap.parse_args()
    args.impl = os.path.abspath(args.impl)
    if args.worker:
        worker(args)
    else:
        driver(args)


if __name__ == "__main__":
    main()
