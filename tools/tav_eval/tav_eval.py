#!/usr/bin/env python3
"""TAV evaluation harness for AlphaEvolve-style search over ``get_tree_model``.

Given a FINN node name and a path to a ``.py`` file that defines a candidate
``get_tree_model`` function, this tool:

1. Splices the candidate ``get_tree_model`` into that node's source file
   (``src/finn/custom_op/fpgadataflow/<node>.py``).
2. Runs the node's analytical-characterization pytest inside the FINN docker
   container (via ``docker exec``), with the rtlsim reference served from cache.
3. Writes a log file -- one line per parametrized test case -- containing the
   decorator parameters, the pass/fail verdict, and the element-wise delta
   between the analytically-modeled token access vector (TAV) and the rtlsim
   reference TAV, for both ports.

The log is stored under the mounted build directory (``$FINN_HOST_BUILD_DIR``,
default ``/tmp/finn_dev_<user>``) which is writable from inside the container,
and its path is returned. With ``-v`` the log contents are also printed.

Usage (CLI):
    python tav_eval.py FMPadding /path/to/candidate_tree_model.py
    python tav_eval.py FMPadding /path/to/candidate.py -v

Usage (import):
    from tav_eval import evaluate_tree_model
    log_path = evaluate_tree_model("FMPadding", "/path/to/candidate.py", verbose=True)
"""

import argparse
import ast
import datetime
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

# location of this harness inside the repo (host paths)
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
FINN_ROOT = os.path.abspath(os.path.join(HARNESS_DIR, "..", ".."))

# ---------------------------------------------------------------------------
# node registry: node name -> source file (relative to FINN_ROOT) + test nodeid
# ``test`` is the primary analytical-characterization test for that node.
# ``extra_tests`` are additional characterization tests exercising the same
# get_tree_model (optional).
# ---------------------------------------------------------------------------
NODE_REGISTRY = {
    "FMPadding": {
        "src": "src/finn/custom_op/fpgadataflow/fmpadding.py",
        "test": "tests/fpgadataflow/test_fpgadataflow_fmpadding.py::"
        "test_fpgadataflow_analytical_characterization_fmpadding",
    },
    "ConvolutionInputGenerator": {
        "src": "src/finn/custom_op/fpgadataflow/convolutioninputgenerator.py",
        "test": "tests/fpgadataflow/test_fpgadataflow_convinputgenerator.py::"
        "test_fpgadataflow_analytical_characterization_slidingwindow",
        "extra_tests": [
            "tests/fpgadataflow/test_fpgadataflow_convinputgenerator.py::"
            "test_fpgadataflow_analytical_characterization_slidingwindow_mobilenet",
            "tests/fpgadataflow/test_fpgadataflow_downsampler.py::"
            "test_fpgadataflow_analytical_characterization_downsampler",
        ],
    },
    "LabelSelect": {
        "src": "src/finn/custom_op/fpgadataflow/labelselect.py",
        "test": "tests/fpgadataflow/test_fpgadataflow_labelselect.py::"
        "test_fpgadataflow_analytical_characterization_labelselect",
    },
    "Thresholding": {
        "src": "src/finn/custom_op/fpgadataflow/thresholding.py",
        "test": "tests/fpgadataflow/test_fpgadataflow_thresholding.py::"
        "test_fpgadataflow_analytical_characterization_thresholding",
    },
    "StreamingDataWidthConverter": {
        "src": "src/finn/custom_op/fpgadataflow/streamingdatawidthconverter.py",
        "test": "tests/fpgadataflow/test_fpgadataflow_dwc.py::"
        "test_fpgadataflow_analytical_characterization_dwc",
    },
    "MVAU": {
        "src": "src/finn/custom_op/fpgadataflow/matrixvectoractivation.py",
        "test": "tests/fpgadataflow/test_fpgadataflow_mvau.py::"
        "test_fpgadataflow_analytical_characterization_mvau",
    },
    "VVAU": {
        "src": "src/finn/custom_op/fpgadataflow/vectorvectoractivation.py",
        "test": "tests/fpgadataflow/test_fpgadataflow_vvau.py::"
        "test_fpgadataflow_analytical_characterization_vvau",
    },
    "Pool": {
        "src": "src/finn/custom_op/fpgadataflow/pool.py",
        "test": "tests/fpgadataflow/test_convert_to_hw_pool_batch.py::"
        "test_analytical_characterization_pool",
    },
}

# convenient aliases (lower-cased lookups also work, see resolve_node)
ALIASES = {
    "fmpadding_rtl": "FMPadding",
    "fmpadding_hls": "FMPadding",
    "convinputgenerator": "ConvolutionInputGenerator",
    "slidingwindow": "ConvolutionInputGenerator",
    "downsampler": "ConvolutionInputGenerator",
    "dwc": "StreamingDataWidthConverter",
    "streamingdatawidthconverter_rtl": "StreamingDataWidthConverter",
    "matrixvectoractivation": "MVAU",
    "vectorvectoractivation": "VVAU",
    "thresholding_rtl": "Thresholding",
}


def resolve_node(node, src_override=None, test_override=None):
    """Map a node name to its source file and test nodeid, honoring overrides."""
    entry = None
    if node in NODE_REGISTRY:
        entry = dict(NODE_REGISTRY[node])
    else:
        key = ALIASES.get(node.lower())
        if key is None:
            # case-insensitive match against registry keys
            for k in NODE_REGISTRY:
                if k.lower() == node.lower():
                    key = k
                    break
        if key is not None:
            entry = dict(NODE_REGISTRY[key])

    if entry is None and not (src_override and test_override):
        raise SystemExit(
            f"Unknown node '{node}'. Known nodes: {', '.join(sorted(NODE_REGISTRY))}.\n"
            f"For an unlisted node, pass both --src and --test."
        )
    if entry is None:
        entry = {}
    if src_override:
        entry["src"] = src_override
    if test_override:
        entry["test"] = test_override
    if "src" not in entry or "test" not in entry:
        raise SystemExit(f"Incomplete mapping for node '{node}'; pass --src and --test.")
    return entry


# ---------------------------------------------------------------------------
# source splicing
# ---------------------------------------------------------------------------
def _find_func_span(tree, func_name):
    """Return (start_line, end_line, node) for the first FunctionDef named
    func_name, 1-indexed and inclusive of any decorators."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = node.lineno
            if node.decorator_list:
                start = min(d.lineno for d in node.decorator_list)
            return start, node.end_lineno, node
    return None


def replace_function(src_path, donor_path, func_name="get_tree_model"):
    """Replace ``func_name`` in ``src_path`` with the one defined in
    ``donor_path``. Keeps a one-time ``.tav_orig`` backup of the pristine file
    and validates that the result still parses. Returns the backup path."""
    donor_src = open(donor_path).read()
    donor_tree = ast.parse(donor_src)
    dspan = _find_func_span(donor_tree, func_name)
    if dspan is None:
        raise SystemExit(f"No function '{func_name}' found in donor file {donor_path}")
    donor_lines = donor_src.splitlines(keepends=True)
    new_block = textwrap.dedent("".join(donor_lines[dspan[0] - 1 : dspan[1]]))

    tgt_src = open(src_path).read()
    tgt_tree = ast.parse(tgt_src)
    tspan = _find_func_span(tgt_tree, func_name)
    if tspan is None:
        raise SystemExit(f"No function '{func_name}' found in target file {src_path}")
    tgt_lines = tgt_src.splitlines(keepends=True)

    def_line = tgt_lines[tspan[0] - 1]
    indent = def_line[: len(def_line) - len(def_line.lstrip())]
    reindented = "".join(
        (indent + ln if ln.strip() else ln) for ln in new_block.splitlines(keepends=True)
    )
    if not reindented.endswith("\n"):
        reindented += "\n"

    new_tgt = "".join(tgt_lines[: tspan[0] - 1]) + reindented + "".join(tgt_lines[tspan[1] :])
    # validate before writing
    ast.parse(new_tgt)

    backup = src_path + ".tav_orig"
    if not os.path.exists(backup):
        shutil.copy2(src_path, backup)
    with open(src_path, "w") as f:
        f.write(new_tgt)
    return backup


def restore_original(src_path):
    """Restore the pristine source file saved on the first patch, if present."""
    backup = src_path + ".tav_orig"
    if os.path.exists(backup):
        shutil.copy2(backup, src_path)
        os.remove(backup)
        return True
    return False


# ---------------------------------------------------------------------------
# docker / container management
# ---------------------------------------------------------------------------
def _container_name():
    uname = getpass.getuser()
    return ("finn_dev_" + uname).lower()


def _host_build_dir():
    return os.environ.get("FINN_HOST_BUILD_DIR", f"/tmp/{_container_name()}")


def _container_running(name):
    out = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    ).stdout
    return name in out.split()


def _finn_docker_tag():
    """The image tag run-docker.sh would use, but with the unstable ``-dirty``
    suffix stripped. The harness dirties the working tree when it splices in a
    candidate get_tree_model, so we must NOT let the tag float with ``--dirty``
    (otherwise it never matches the image that was built on a clean tree).
    Honors an explicit FINN_DOCKER_TAG override."""
    override = os.environ.get("FINN_DOCKER_TAG")
    if override:
        return override
    describe = subprocess.run(
        ["git", "describe", "--always", "--tags"],
        cwd=FINN_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    xrt = os.environ.get("XRT_DEB_VERSION", "xrt_202220.2.14.354_22.04-amd64-xrt")
    return f"xilinx/finn:{describe}.{xrt}"


def _list_finn_images():
    out = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "xilinx/finn"],
        capture_output=True,
        text=True,
    ).stdout
    return [ln for ln in out.split() if ln and "<none>" not in ln]


def ensure_container(name, timeout=1800):
    """Make sure a long-running FINN container called ``name`` exists and has
    its python dependencies installed. Starts one (detached) if needed.

    Image selection (in order): an existing image matching the stable tag; any
    existing ``xilinx/finn`` image; otherwise build one via run-docker.sh."""
    if not _container_running(name):
        env = dict(os.environ)
        env["FINN_DOCKER_EXTRA"] = (env.get("FINN_DOCKER_EXTRA", "") + f" -d --name {name}").strip()
        tag = _finn_docker_tag()
        images = _list_finn_images()
        if tag in images:
            env["FINN_DOCKER_PREBUILT"] = "1"
            env["FINN_DOCKER_TAG"] = tag
            print(f"[tav_eval] starting container '{name}' from image {tag}", file=sys.stderr)
        elif images:
            env["FINN_DOCKER_PREBUILT"] = "1"
            env["FINN_DOCKER_TAG"] = images[0]
            print(
                f"[tav_eval] starting container '{name}' from existing image {images[0]}",
                file=sys.stderr,
            )
        else:
            # no image available: let run-docker.sh build one, pinned to the
            # stable tag so it is reused on subsequent runs despite the dirty tree
            env["FINN_DOCKER_TAG"] = tag
            env.setdefault("FINN_DOCKER_PREBUILT", "0")
            print(
                f"[tav_eval] no xilinx/finn image found; building {tag} via run-docker.sh "
                "(first run only, this can take a while) ...",
                file=sys.stderr,
            )
        subprocess.run(
            ["./run-docker.sh", "sleep", "infinity"],
            cwd=FINN_ROOT,
            env=env,
            check=True,
        )
    # wait until the FINN python stack is importable (entrypoint installs -e deps)
    print("[tav_eval] waiting for container python environment ...", file=sys.stderr)
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _container_running(name):
            raise SystemExit(f"container '{name}' is not running")
        r = subprocess.run(
            [
                "docker", "exec", "-e", "HOME=/tmp/home_dir", name,
                "bash", "-lc", "python -c 'import finn, qonnx, brevitas'",
            ],
            capture_output=True,
        )
        if r.returncode == 0:
            return
        time.sleep(5)
    raise SystemExit(f"timed out waiting for container '{name}' to become ready")


def sync_cache(cache_dir=None):
    """Copy committed rtlsim reference models into ``$FINN_BUILD_DIR`` so the
    characterization test finds them as a cache hit (and therefore skips
    rtlsim). The reference directories are named after the stringified
    ``node_details`` tuple, e.g.
    ``cached_models/('Downsampler', False, False, 32, 1, 2, 'rtlsim')<suffix>/model_rtlsim.onnx``.
    Existing destination directories are left untouched. Returns the number of
    reference dirs synced."""
    if cache_dir is None:
        cache_dir = os.path.join(FINN_ROOT, "cached_models")
    if not os.path.isdir(cache_dir):
        return 0
    build_dir = _host_build_dir()
    os.makedirs(build_dir, exist_ok=True)
    n = 0
    for entry in os.listdir(cache_dir):
        src = os.path.join(cache_dir, entry)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(build_dir, entry)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
            n += 1
    return n


# docker/finn_entrypoint.sh sources Vitis/Vivado's settings64.sh (which puts
# xelab etc. on PATH) once, in the entrypoint process, right before it execs
# into "sleep infinity". Those exports never reach `docker exec` -- a fresh
# exec only inherits the image's ENV plus whatever was passed via `docker run
# -e`, not env changes a script made at runtime inside the container. So
# rtlsim tools are invisible to `docker exec` sessions unless we re-source the
# same settings64.sh here; this mirrors finn_entrypoint.sh's logic (Vitis
# preferred, falling back to Vivado-only, plus the matching LD_LIBRARY_PATH
# additions) without repeating its one-time pip-install/finn_xsi-build steps.
_XILINX_ENV_SOURCE = textwrap.dedent("""\
    if [ -f "$VITIS_PATH/settings64.sh" ]; then
        export XILINX_VITIS="$VITIS_PATH"
        export XILINX_XRT="${XILINX_XRT:-/opt/xilinx/xrt}"
        source "$VITIS_PATH/settings64.sh"
        if [ -f "$XILINX_XRT/setup.sh" ]; then
            source "$XILINX_XRT/setup.sh"
        fi
    elif [ -f "$VIVADO_PATH/settings64.sh" ]; then
        export XILINX_VIVADO="$VIVADO_PATH"
        source "$VIVADO_PATH/settings64.sh"
    fi
    if [ -n "$XILINX_VIVADO" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/:${XILINX_VIVADO}/lib/lnx64.o"
    fi
    if [ -f "$HLS_PATH/settings64.sh" ]; then
        source "$HLS_PATH/settings64.sh"
    fi
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$VITIS_PATH/lnx64/tools/fpo_v7_1:$HLS_PATH/lnx64/tools/fpo_v7_1"
""")


def run_pytest(name, test_nodeids, records_dir, pytest_log):
    """Run the given pytest nodeids inside the container, capturing the plugin
    JSON records into ``records_dir`` and the raw pytest output into
    ``pytest_log``. Returns the pytest exit code."""
    build_dir = _host_build_dir()  # mounted at the same path inside the container
    plugin_dir = os.path.join(FINN_ROOT, "tools", "tav_eval")
    inner = (
        f"{_XILINX_ENV_SOURCE}"
        f"cd {FINN_ROOT} && "
        f"python -m pytest {' '.join(repr(t) for t in test_nodeids)} "
        f"-p _tav_eval_plugin -p no:cacheprovider -o addopts='' -rA -v"
    )
    # run the exec as the same uid:gid run-docker.sh used to start the container,
    # so files written into the build dir stay owned by the host user
    try:
        uid_gid = f"{os.getuid()}:{os.getgid()}"
    except AttributeError:  # pragma: no cover - non-POSIX
        uid_gid = None
    cmd = [
        "docker", "exec",
        "-e", "HOME=/tmp/home_dir",
        "-e", f"FINN_BUILD_DIR={build_dir}",
        "-e", f"TAV_EVAL_OUT={records_dir}",
        "-e", f"PYTHONPATH={plugin_dir}",
    ]
    if uid_gid:
        cmd += ["--user", uid_gid]
    cmd += [name, "bash", "-lc", inner]
    with open(pytest_log, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


# ---------------------------------------------------------------------------
# log assembly
# ---------------------------------------------------------------------------
def _fmt_params(params):
    return " ".join(f"{k}={v}" for k, v in params.items())


def _fmt_vec(vec, limit=64):
    if len(vec) > limit:
        head = ", ".join(str(x) for x in vec[:limit])
        return f"[{head}, ... (+{len(vec) - limit} more)]"
    return "[" + ", ".join(str(x) for x in vec) + "]"


def _verdict_tag(rec):
    outcome = rec.get("outcome")
    if outcome == "passed":
        return "PASS"
    if outcome == "skipped":
        return "SKIP"
    if outcome == "failed":
        # distinguish a clean TAV mismatch from an execution error
        if rec.get("ports"):
            return "FAIL"
        return "ERROR"
    return outcome.upper() if outcome else "UNKNOWN"


def load_records(records_dir):
    """Load the per-test JSON records produced by the plugin, sorted by nodeid."""
    records = []
    if os.path.isdir(records_dir):
        for fn in sorted(os.listdir(records_dir)):
            if fn.endswith(".json"):
                try:
                    records.append(json.load(open(os.path.join(records_dir, fn))))
                except Exception:
                    pass
    records.sort(key=lambda r: r.get("nodeid", ""))
    return records


def score_records(records):
    """Reduce a set of records to an optimization fitness (lower is better).

    score = sum over non-skipped cases of (peak_volume_delta + |len_delta|) for
    both ports; ERROR cases (no comparison happened) get a large penalty so the
    optimizer avoids broken candidates. score == 0 with no fails/errors means
    the analytical TAVs match the rtlsim references exactly."""
    ERROR_PENALTY = 1_000_000
    score = 0.0
    n_pass = n_fail = n_skip = n_error = 0
    for r in records:
        outcome = r.get("outcome")
        if outcome == "skipped":
            n_skip += 1
            continue
        ports = r.get("ports")
        if not ports:
            n_error += 1
            score += ERROR_PENALTY
            continue
        if outcome == "passed":
            n_pass += 1
        else:
            n_fail += 1
        for p in ports:
            score += abs(p.get("peak_volume_delta", 0)) + abs(p.get("len_delta", 0))
    return {
        "score": score,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "n_error": n_error,
        "n_total": len(records),
        "solved": n_fail == 0 and n_error == 0 and score == 0,
    }


def assemble_log(records_dir, pytest_log, out_log, header):
    records = load_records(records_dir)

    lines = []
    lines.extend(header)
    lines.append("")

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    for rec in records:
        tag = _verdict_tag(rec)
        counts[tag] = counts.get(tag, 0) + 1
        params = _fmt_params(rec.get("params", {}))
        node = rec.get("params", {})  # noqa
        parts = [f"[{tag}]", rec.get("nodeid", "?")]
        if params:
            parts.append("| " + params)
        if rec.get("ports"):
            for p in rec["ports"]:
                parts.append(
                    f"| {p['port']}: len_a={p['len_analytical']} "
                    f"len_rtl={p['len_rtlsim']} len_delta={p['len_delta']} "
                    f"peak={p['peak_volume_delta']} delta={_fmt_vec(p['delta_vector'])}"
                )
        elif rec.get("longrepr"):
            parts.append("| " + rec["longrepr"].splitlines()[-1])
        lines.append(" ".join(parts))

    lines.append("")
    lines.append(
        "SUMMARY: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
        + f"  (total {sum(counts.values())} cases)"
    )
    if not records:
        lines.append(
            "WARNING: no characterization records were produced. "
            f"See raw pytest output: {pytest_log}"
        )
    lines.append(f"raw pytest output: {pytest_log}")

    text = "\n".join(lines) + "\n"
    with open(out_log, "w") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------
def evaluate_tree_model(
    node,
    tree_model_path,
    verbose=False,
    src_override=None,
    test_override=None,
    include_extra_tests=False,
    func_name="get_tree_model",
    container=None,
    cache_dir=None,
    quiet=False,
    return_records=False,
):
    """Splice the candidate ``get_tree_model`` from ``tree_model_path`` into the
    given ``node``'s source, run its characterization pytest in docker, and
    write a per-case TAV-delta log into the mounted build directory.

    Returns the absolute path of the log file (or, if ``return_records`` is
    True, a ``(log_path, records)`` tuple where ``records`` is the list of
    per-test JSON records). If ``verbose`` is True, the log contents are also
    printed to stdout; ``quiet`` suppresses the log-path stdout line (useful in
    a loop)."""
    tree_model_path = os.path.abspath(tree_model_path)
    if not os.path.isfile(tree_model_path):
        raise SystemExit(f"tree-model file not found: {tree_model_path}")

    entry = resolve_node(node, src_override, test_override)
    src_path = entry["src"]
    if not os.path.isabs(src_path):
        src_path = os.path.join(FINN_ROOT, src_path)
    test_nodeids = [entry["test"]]
    if include_extra_tests:
        test_nodeids += entry.get("extra_tests", [])

    name = container or _container_name()

    # 1. splice in the candidate get_tree_model
    backup = replace_function(src_path, tree_model_path, func_name)
    print(f"[tav_eval] patched {func_name} in {src_path} (backup: {backup})", file=sys.stderr)

    # 2. make sure the container is up and the rtlsim reference cache is in place
    ensure_container(name)
    synced = sync_cache(cache_dir)
    print(f"[tav_eval] synced {synced} cached rtlsim reference(s) into build dir", file=sys.stderr)

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_node = re.sub(r"[^A-Za-z0-9_.-]", "_", node)
    base = os.path.join(_host_build_dir(), "tav_eval", f"{safe_node}-{run_id}")
    records_dir = os.path.join(base, "records")
    os.makedirs(records_dir, exist_ok=True)
    pytest_log = os.path.join(base, "pytest.out")
    out_log = os.path.join(base, "tav_eval.log")

    rc = run_pytest(name, test_nodeids, records_dir, pytest_log)
    print(f"[tav_eval] pytest exit code: {rc}", file=sys.stderr)

    header = [
        "# TAV evaluation log",
        f"# node:        {node}",
        f"# source:      {src_path}",
        f"# tree_model:  {tree_model_path}",
        f"# tests:       {', '.join(test_nodeids)}",
        f"# timestamp:   {run_id}",
        f"# pytest_rc:   {rc}",
    ]
    text = assemble_log(records_dir, pytest_log, out_log, header)

    if not quiet:
        print(out_log)
    if verbose:
        print(text)
    if return_records:
        return out_log, load_records(records_dir)
    return out_log


def _build_argparser():
    p = argparse.ArgumentParser(
        description="Splice a candidate get_tree_model into a FINN node and "
        "evaluate it against the rtlsim TAV reference via pytest in docker."
    )
    p.add_argument("node", nargs="?", help="node name, e.g. FMPadding (see --list)")
    p.add_argument("tree_model_path", nargs="?", help="path to .py file defining get_tree_model")
    p.add_argument("-v", "--verbose", action="store_true", help="also print the log contents")
    p.add_argument("--src", help="override the node source file path")
    p.add_argument("--test", help="override the pytest nodeid to run")
    p.add_argument(
        "--extra-tests", action="store_true", help="also run any extra characterization tests"
    )
    p.add_argument("--func-name", default="get_tree_model", help="function name to replace")
    p.add_argument("--container", help="docker container name (default finn_dev_<user>)")
    p.add_argument(
        "--cache-dir",
        help="dir of committed rtlsim reference models to sync into the build dir "
        "(default <repo>/cached_models)",
    )
    p.add_argument("--list", action="store_true", help="list known nodes and exit")
    p.add_argument(
        "--restore",
        action="store_true",
        help="restore the pristine node source (from the .tav_orig backup) and exit",
    )
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    if args.list:
        for k, v in sorted(NODE_REGISTRY.items()):
            print(f"{k:30s} {v['src']}")
        return 0
    if not args.node:
        raise SystemExit("node is required (see --list)")
    if args.restore:
        entry = resolve_node(args.node, args.src, args.test)
        src_path = entry["src"]
        if not os.path.isabs(src_path):
            src_path = os.path.join(FINN_ROOT, src_path)
        ok = restore_original(src_path)
        print(f"{'restored' if ok else 'no backup found for'} {src_path}")
        return 0
    if not args.tree_model_path:
        raise SystemExit("tree_model_path is required (path to .py with get_tree_model)")
    evaluate_tree_model(
        args.node,
        args.tree_model_path,
        verbose=args.verbose,
        src_override=args.src,
        test_override=args.test,
        include_extra_tests=args.extra_tests,
        func_name=args.func_name,
        container=args.container,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
