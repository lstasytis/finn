"""Iteration loop for FINN get_tree_model search, evaluated by tav_eval.

Same generate -> run -> evaluate -> reprompt shape as iteration_loop.py: the
agent writes `get_tree_model.py` into the workspace; check_output() here
splices it into the target FINN node (via tools/tav_eval) and runs the node's
analytical-characterization pytest in the FINN docker container, comparing the
analytical token access vector (TAV) against the rtlsim reference. The
per-case delta feedback drives the next prompt. Stops when every case matches
the reference exactly, or after --max-iterations.

The TASK prompt below is a starting point -- tune it for whatever guidance
gets the model to converge faster; the evaluation side (check_output) doesn't
need to change when you do.

Run:
    python examples/tav_tree_model_loop.py ConvolutionInputGenerator
    python examples/tav_tree_model_loop.py FMPadding --model gpt-5.1 --max-iterations 10
    python examples/tav_tree_model_loop.py ConvolutionInputGenerator --apply-best
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from agent_stub.agent import run_agent
from agent_stub.models import DEFAULT_MODEL

# tav_eval lives one level up in the FINN repo (tools/tav_eval), not inside
# this package -- add it to sys.path rather than vendoring a copy.
_TAV_EVAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tav_eval")
)
if _TAV_EVAL_DIR not in sys.path:
    sys.path.insert(0, _TAV_EVAL_DIR)
import tav_eval  # noqa: E402

CANDIDATE_FILENAME = "get_tree_model.py"

# ── per-node HLS/RTL reference pointers ─────────────────────────────────────
# Filled into the task prompt below so the agent knows where to go read the
# node's "ground truth" hardware behaviour. Paths are relative to the FINN
# repo root (given to the agent separately). Kept here (not in tav_eval) since
# it's prompt content, not evaluation logic.
NODE_REFS = {
    "FMPadding": {
        "hls": "(no HLS backend for this node -- it is RTL-only)",
        "rtl": "finn-rtllib/fmpadding/hdl/fmpadding_axi.sv, fmpadding.sv, axi2we.sv "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/rtl/fmpadding_rtl.py)",
    },
    "ConvolutionInputGenerator": {
        "hls": "(no HLS backend for this node -- it is RTL-only)",
        "rtl": "finn-rtllib/swg/swg_template_default.sv, swg_template_default_dynamic.sv, "
        "swg_template_parallel.sv, swg_common.sv, swg_pkg.sv "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/rtl/convolutioninputgenerator_rtl.py)",
    },
    "LabelSelect": {
        "hls": "deps/finn-hlslib/maxpool.h, function LabelSelect_Batch "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/hls/labelselect_hls.py)",
        "rtl": "(no RTL backend for this node -- it is HLS-only)",
    },
    "Thresholding": {
        "hls": "deps/finn-hlslib/activations.hpp, function Thresholding_Batch "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py)",
        "rtl": "finn-rtllib/thresholding/hdl/thresholding.sv, thresholding_axi.sv "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py)",
    },
    "StreamingDataWidthConverter": {
        "hls": "deps/finn-hlslib/streamtools.h, function StreamingDataWidthConverter_Batch "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/hls/streamingdatawidthconverter_hls.py)",
        "rtl": "finn-rtllib/dwc/hdl/dwc.sv, dwc_axi.sv "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/rtl/streamingdatawidthconverter_rtl.py)",
    },
    "MVAU": {
        "hls": "deps/finn-hlslib/mvau.hpp, function Matrix_Vector_Activate_Batch "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/hls/matrixvectoractivation_hls.py)",
        "rtl": "finn-rtllib/mvu/mvu_pkg.sv, mvu_vvu_axi.sv, mvu.sv, mvu_vvu_8sx9_dsp58.sv "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py)",
    },
    "VVAU": {
        "hls": "deps/finn-hlslib/vvau.hpp, function Vector_Vector_Activate_Batch "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/hls/vectorvectoractivation_hls.py)",
        "rtl": "finn-rtllib/mvu/mvu_pkg.sv, mvu_vvu_axi.sv, mvu.sv, mvu_vvu_8sx9_dsp58.sv "
        "(same RTL engine as MVAU; Python wrapper: "
        "src/finn/custom_op/fpgadataflow/rtl/vectorvectoractivation_rtl.py)",
    },
    "Pool": {
        "hls": "deps/finn-hlslib/pool.hpp, function Pool_batch "
        "(Python wrapper: src/finn/custom_op/fpgadataflow/hls/pool_hls.py)",
        "rtl": "(no RTL backend for this node -- it is HLS-only)",
    },
}


def _node_refs(node, src_path):
    refs = NODE_REFS.get(node)
    if refs is None:
        return (
            "(unlisted node -- inspect src/finn/custom_op/fpgadataflow/hls/ and "
            "src/finn/custom_op/fpgadataflow/rtl/ for the matching backend file)",
            "(unlisted node -- see above)",
        )
    return refs["hls"], refs["rtl"]


# ── prompt ───────────────────────────────────────────────────────────────────
# This is the part meant to be tuned by hand; check_output() below does not
# depend on its wording.
TASK_HEADER = textwrap.dedent("""\
    You are an FPGA expert in Vitis HLS and SystemVerilog, designing
    characteristic tree models of ML operators described in this repo's
    deps/finn-hlslib and finn-rtllib directories. You are optimizing node
    {node}. The existing tree is in {src_path}'s get_tree_model() function.
    HLS reference: {hls_ref}. RTL reference: {rtl_ref}.

    Each candidate tree is executed to produce a token access vector (TAV),
    compared against an rtl-simulated ground truth. Goal: make the tree
    produce a TAV identical to rtlsim across all testcases.

    TAVs come from Characteristic_Node (src/finn/util/basic.py). Each
    Characteristic_Node holds a list of states (tree nodes) plus how many
    times each state repeats (edge values). A leaf node is a tuple flagging
    whether its state reads, writes, both, or neither -- each flag adds +1 to
    the read/write vector at that clock cycle. The tree is effectively a
    cycle-accurate model of the node, but the only thing it needs to capture
    is input/output channel activity (reads/writes), not full datapath
    behavior.

    Approach: extract the node's compile-time parameters via
    self.get_nodeattr(...) -- these determine which states exist and their
    repeat counts, and should be encoded into the tree's edges. Look at other
    nodes' trees in src/finn/custom_op for examples (RTL and HLS backends
    typically need distinct trees, since their cycle behavior differs). The
    most common mistake is misjudging when a read and a write overlap in the
    same cycle.

    Build incrementally: first get correct volume (total tokens read/written
    matches rtlsim), then correct length (fuse/split phases or add idle
    states so cycle count matches), then exact equality (fusing reads/writes
    and partial states correctly, cycle by cycle).

    You should not attempt to simulate the node internally with for loops,
    you are trying to determine the unique states that make up the node being modelled.
    The tree is used to generate the TAV by traversing each node recursively and upon
    hitting a leaf, append 1 value to the TAV (effectively progressing the node's cycle counter by 1.
    If the leaf contains a value 1, that means that either the input (if its the first value) or the output (if its the second),
    value being appended (a counter of total tokens read or written) is incremented by one.

    When the test is executed, we produce a TAV by traversing the tree and compare to a ground truth rtlsim result.
    The vectors produced need to become identical. This is possible as the number of unique states is limited,
    you should need to use more than 10-40 nodes to simulate any node. You should perform the tree construction systematically,
    tree nodes should be described similarly to how they are in the current tree models, starting with a root node and working 
    downwards to more fine-grain stages of the operator's execution.


    The FINN repository is checked out at {finn_root} -- you may read any
    file in it (including the hls/rtl sources above and {src_path} itself)
    with the bash tool; only writes are confined to your workspace.

    Write your tree model as a file named `{filename}` in the workspace,
    containing exactly one top-level function:

        def get_tree_model(self):
            ...
            return <top-level Characteristic_Node>

    This file is spliced directly into the node's module in place of its
    existing get_tree_model -- do not write import statements; you may use
    any name already available in that module (e.g. Characteristic_Node,
    math, np, self.get_nodeattr(...)). Don't try to call the function
    yourself (there is no real `self` outside the node); just write it and
    make sure it's syntactically valid.

    Reference -- the node's current get_tree_model:
    ```python
    {baseline}
    ```
""")

RETRY_SUFFIX = textwrap.dedent("""

    Your previous attempt:
    ```python
    {previous}
    ```

    Evaluation feedback (delta = analytical - rtlsim; zero everywhere is the
    goal):
    {feedback}

    Write an improved `{filename}` that reduces these deltas.
""")


def build_task(node, src_path, baseline, previous=None, feedback=None):
    hls_ref, rtl_ref = _node_refs(node, src_path)
    task = TASK_HEADER.format(
        node=node,
        src_path=src_path,
        hls_ref=hls_ref,
        rtl_ref=rtl_ref,
        finn_root=tav_eval.FINN_ROOT,
        filename=CANDIDATE_FILENAME,
        baseline=baseline.strip(),
    )
    if previous is not None:
        task += RETRY_SUFFIX.format(
            previous=previous.strip(), feedback=feedback, filename=CANDIDATE_FILENAME
        )
    return task


# ── evaluation (the part that matters -- wires tav_eval in as the evaluator)─
def _format_feedback(records, max_cases=12, vec_preview=24):
    lines = []
    for r in records[:max_cases]:
        tag = tav_eval._verdict_tag(r)
        params = tav_eval._fmt_params(r.get("params", {}))
        if r.get("ports"):
            bits = []
            for p in r["ports"]:
                vec = p["delta_vector"][:vec_preview]
                more = len(p["delta_vector"]) - len(vec)
                vec_s = "[" + ", ".join(str(x) for x in vec) + (f", +{more}...]" if more > 0 else "]")
                bits.append(f"{p['port']} peak={p['peak_volume_delta']} len_delta={p['len_delta']} delta={vec_s}")
            lines.append(f"  [{tag}] {params} | " + " | ".join(bits))
        else:
            tail = (r.get("longrepr") or "").splitlines()
            lines.append(f"  [{tag}] {params} | {tail[-1] if tail else ''}")
    more = len(records) - max_cases
    if more > 0:
        lines.append(f"  ... +{more} more case(s)")
    return "\n".join(lines)


def check_output(
    workspace,
    node,
    *,
    src_override=None,
    test_override=None,
    include_extra_tests=False,
    cache_dir=None,
):
    """Return (passed, feedback_for_next_prompt, records, score).

    Splices workspace/get_tree_model.py into the node's source and runs its
    characterization pytest via tav_eval; this is the evaluator swapped into
    Bespoke's generate -> evaluate -> reprompt loop."""
    candidate = workspace / CANDIDATE_FILENAME
    if not candidate.exists():
        return False, f"{CANDIDATE_FILENAME} was not created in the workspace.", [], None

    try:
        _, records = tav_eval.evaluate_tree_model(
            node,
            str(candidate),
            src_override=src_override,
            test_override=test_override,
            include_extra_tests=include_extra_tests,
            cache_dir=cache_dir,
            quiet=True,
            return_records=True,
        )
    except SystemExit as e:
        return False, f"tav_eval could not evaluate the candidate: {e}", [], None

    sc = tav_eval.score_records(records)
    if sc["solved"]:
        return True, "", records, sc
    feedback = (
        f"score={sc['score']} (pass={sc['n_pass']} fail={sc['n_fail']} "
        f"error={sc['n_error']} skip={sc['n_skip']})\n" + _format_feedback(records)
    )
    return False, feedback, records, sc


# ── main loop ────────────────────────────────────────────────────────────────
def _baseline_path(node, src_override=None, test_override=None):
    entry = tav_eval.resolve_node(node, src_override, test_override)
    base = os.path.basename(entry["src"]).replace(".py", "_tree_model.py")
    return os.path.join(_TAV_EVAL_DIR, "examples", base)


DEFAULT_LOG_PATH = "llm_tree_modeling.log"


def _make_logger(log_path):
    """Open log_path (truncated) and return a log(msg) that both prints and
    appends a timestamped line, flushed immediately so it can be tailed."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", buffering=1)

    def log(msg=""):
        print(msg)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        for line in msg.splitlines() or [""]:
            fh.write(f"[{ts}] {line}\n")
        fh.flush()

    return log, fh


def run_loop(
    node,
    model=DEFAULT_MODEL,
    workspace=Path("workspace"),
    max_iterations=10,
    max_turns=30,
    baseline=None,
    src_override=None,
    test_override=None,
    include_extra_tests=False,
    cache_dir=None,
    apply_best=False,
    out_dir=None,
    log_path=DEFAULT_LOG_PATH,
):
    log, log_fh = _make_logger(log_path)
    try:
        entry = tav_eval.resolve_node(node, src_override, test_override)
        src_path = entry["src"]

        baseline = baseline or _baseline_path(node, src_override, test_override)
        if not os.path.isfile(baseline):
            raise SystemExit(f"baseline candidate not found: {baseline}")

        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_node = re.sub(r"[^A-Za-z0-9_.-]", "_", node)
        out_dir = Path(out_dir or os.path.join(
            tav_eval._host_build_dir(), "tav_bespoke", f"{safe_node}-{run_id}"
        ))
        cand_dir = out_dir / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)

        workspace.mkdir(parents=True, exist_ok=True)
        candidate_path = workspace / CANDIDATE_FILENAME

        baseline_src = Path(baseline).read_text()
        best = {"iteration": 0, "source": baseline_src, "score": float("inf")}
        history = []

        log(f"node={node} model={model} max_iterations={max_iterations} out_dir={out_dir}")

        previous, feedback = None, None
        for i in range(1, max_iterations + 1):
            log(f"\n{'=' * 60}\niteration {i}\n{'=' * 60}")
            task = build_task(node, src_path, baseline_src, previous=previous, feedback=feedback)
            run_agent(task, model, workspace, max_turns=max_turns)

            passed, feedback, records, sc = check_output(
                workspace, node,
                src_override=src_override, test_override=test_override,
                include_extra_tests=include_extra_tests, cache_dir=cache_dir,
            )
            previous = candidate_path.read_text() if candidate_path.exists() else None

            if previous is not None:
                (cand_dir / f"iter_{i:03d}.py").write_text(previous)
            if sc is not None:
                log(f"\niter {i}: score={sc['score']} (pass={sc['n_pass']} fail={sc['n_fail']} "
                    f"error={sc['n_error']})")
                history.append({"iteration": i, **sc})
                if sc["score"] < best["score"] and previous is not None:
                    best = {"iteration": i, "source": previous, "score": sc["score"]}
            else:
                history.append({"iteration": i, "error": feedback})

            if passed:
                log(f"\nAll cases matched the rtlsim reference on iteration {i}.")
                break
            log(f"\nfeedback:\n{feedback}")
        else:
            log(f"\nStopped after {max_iterations} iterations.")

        best_path = out_dir / "best_get_tree_model.py"
        best_path.write_text(best["source"])
        (out_dir / "history.json").write_text(json.dumps({"node": node, "best": best["iteration"],
                                                            "best_score": best["score"],
                                                            "history": history}, indent=1))

        if not os.path.isabs(src_path):
            src_path = os.path.join(tav_eval.FINN_ROOT, src_path)
        tav_eval.restore_original(src_path)
        if apply_best:
            tav_eval.replace_function(src_path, str(best_path))
            log(f"applied best candidate (iteration {best['iteration']}) to {src_path}")

        log(f"\nbest candidate: {best_path}")
        log(f"history:        {out_dir / 'history.json'}")
        return best_path
    finally:
        log_fh.close()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", help="FINN node name, e.g. ConvolutionInputGenerator (tav_eval.py --list)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=30, help="agent tool-call turns per iteration")
    ap.add_argument("--baseline", help="seed get_tree_model.py (default: tav_eval's example for this node)")
    ap.add_argument("--src", help="override the node source file path")
    ap.add_argument("--test", help="override the pytest nodeid to run")
    ap.add_argument("--extra-tests", action="store_true", help="also run registry extra_tests")
    ap.add_argument("--cache-dir", help="rtlsim reference cache dir (default <repo>/cached_models)")
    ap.add_argument("--out", help="output directory (default under $FINN_HOST_BUILD_DIR/tav_bespoke)")
    ap.add_argument("--apply-best", action="store_true",
                     help="splice the best candidate into the node source at the end")
    ap.add_argument("--log", default=DEFAULT_LOG_PATH,
                     help=f"log file to tail while the loop runs (default: {DEFAULT_LOG_PATH})")
    args = ap.parse_args()

    run_loop(
        args.node,
        model=args.model,
        workspace=Path(args.workspace).resolve(),
        max_iterations=args.max_iterations,
        max_turns=args.max_turns,
        baseline=args.baseline,
        src_override=args.src,
        test_override=args.test,
        include_extra_tests=args.extra_tests,
        log_path=args.log,
        cache_dir=args.cache_dir,
        apply_best=args.apply_best,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
