"""Iteration loop for FINN get_tree_model search, evaluated by tav_eval.

Same generate -> run -> evaluate -> reprompt shape as iteration_loop.py: the
agent writes `get_tree_model.py` into the workspace; check_output() here
splices it into the target FINN node (via tools/tav_eval) and runs the node's
analytical-characterization pytest in the FINN docker container, comparing the
analytical token access vector (TAV) against the rtlsim reference. The
per-case delta feedback drives the next prompt. Stops when every case matches
the reference exactly, or after --max-iterations.

check_output() also runs a second "analyzer" agent (same model, its own
fresh context per iteration) that reviews the candidate and feedback and
proposes structural fixes in plain text; its reply is appended to the
feedback the tree-generating agent sees next.

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
import traceback
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
    deps/finn-hlslib directory (HLS) and finn-rtllib directory at the repo
    root (RTL, not under deps/). You are optimizing node
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

    Feedback reports each test case's input and output ports as separate
    pass/fail cases -- you don't need both sides right at once. It's
    usually easier to get one port (often input) passing across all cases
    first, then build on that working tree to additionally get the other
    port correct, rather than trying to fix both simultaneously.

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

    Since the file is small and tends to change substantially between
    attempts, prefer rewriting it wholesale with `*** Add File: {filename}`
    plus the full new contents -- it overwrites unconditionally whether or
    not the file already exists, so there's no patch context to get wrong.
    For `*** Add File: {filename}`, the body is just the raw file contents,
    one line per line -- start directly with `def get_tree_model(self):` on
    the first body line. Do not prepend a `+++` (or `---`) marker line of
    any kind, even a bare one with nothing after it; the tool does not use
    that convention and a stray marker line will corrupt the file.
    If you do use `*** Update File: {filename}` for a small targeted edit,
    follow the format the tool describes exactly (context lines, '-'/'+',
    bare '@@ anchor' lines) -- do not emit git-style unified-diff headers
    ('--- file', '+++ file', or '@@ -a,b +c,d @@' line-count headers), they
    are not supported and will fail to apply.

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

    Evaluation feedback (delta = analytical - rtlsim per cycle, zero
    everywhere is the goal). Each port's full delta vector is run-length
    encoded as a lossless list of (run_length, value) pairs, e.g. (480,-1)
    means the next 480 cycles are each off by -1; (1,0),(20,-1) means one
    matching cycle then 20 cycles off by -1. You are aiming for each port's
    encoding to collapse to a single pair, (vector_length,0), meaning no
    deltas anywhere:
    {feedback}

    Write an improved `{filename}` that reduces these deltas.
""")

# ── second agent: analysis only, no tree-writing ────────────────────────────
# Reviews the candidate plus this and the previous iteration's feedback in
# its own fresh context and proposes concrete structural fixes; its reply is
# appended to the feedback the tree-generating agent sees next (see
# check_output below), it never edits the candidate itself.
ANALYSIS_TASK = textwrap.dedent("""\
    You are an FPGA expert in Vitis HLS and SystemVerilog, reviewing a
    candidate characteristic tree model for node {node}, used by
    Characteristic_Node (src/finn/util/basic.py) to produce a token access
    vector (TAV) of the node's input/output channel read/write activity,
    compared against an rtl-simulated ground truth. HLS reference:
    {hls_ref}. RTL reference: {rtl_ref}. The FINN repository is checked out
    at {finn_root} -- you may read any file in it with the bash tool to
    check these references.

    You are NOT generating or editing the tree yourself -- a separate agent
    does that. Your only job is to analyze the candidate below against its
    evaluation feedback and write concrete, actionable suggestions for how
    the tree's structure should change (e.g. a missing state, a wrong
    repeat count, a misjudged read/write overlap, a phase that should be
    fused or split) to close the remaining gaps. Reply with your analysis
    as plain text -- do not use apply_patch or write any files.

    Candidate `{filename}` under test:
    ```python
    {candidate}
    ```

    Evaluation feedback (delta = analytical - rtlsim per cycle). Each
    port's full delta vector is run-length encoded as a lossless list of
    (run_length, value) pairs, e.g. (480,-1) means the next 480 cycles are
    each off by -1; (1,0),(20,-1) means one matching cycle then 20 cycles
    off by -1. You are aiming for each port's encoding to collapse to a
    single pair, (vector_length,0), meaning no deltas anywhere:
    {feedback}
    {previous_block}""")

PREVIOUS_FEEDBACK_BLOCK = textwrap.dedent("""
    For comparison, here is the previous iteration's evaluation feedback
    (one step back only -- not the full history):
    {previous_feedback}
""")


def build_analysis_task(node, candidate_source, feedback, previous_feedback=None):
    hls_ref, rtl_ref = _node_refs(node, None)
    previous_block = (
        PREVIOUS_FEEDBACK_BLOCK.format(previous_feedback=previous_feedback)
        if previous_feedback else ""
    )
    return ANALYSIS_TASK.format(
        node=node,
        hls_ref=hls_ref,
        rtl_ref=rtl_ref,
        finn_root=tav_eval.FINN_ROOT,
        filename=CANDIDATE_FILENAME,
        candidate=candidate_source.strip(),
        feedback=feedback,
        previous_block=previous_block,
    )


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
def _rle(vec):
    """Lossless run-length encoding of an integer sequence into a list of
    (run_length, value) pairs, e.g. [1, 1, -1, -1, -1] -> [(2, 1), (3, -1)]."""
    pairs = []
    for v in vec:
        if pairs and pairs[-1][1] == v:
            pairs[-1] = (pairs[-1][0] + 1, v)
        else:
            pairs.append((1, v))
    return pairs


def _fmt_rle(vec):
    return ", ".join(f"({n},{v:+d})" if v else f"({n},0)" for n, v in _rle(vec))


# Some characterization tests parametrize over a single packed "config"
# tuple instead of individually named arguments (see e.g. config = (shape,
# inWidth, outWidth, finn_dtype) in test_fpgadataflow_dwc.py) -- the LLM
# would otherwise have to reverse-engineer the tuple order from the test
# file, so decode known ones into named fields here.
_CONFIG_TUPLE_FIELDS = {
    "StreamingDataWidthConverter": ("shape", "inWidth", "outWidth", "dataType"),
}


def _split_top_level(s):
    """Split the inside of a tuple/list repr on top-level commas, respecting
    nested brackets, e.g. "[1, 24], 8, INT2" -> ["[1, 24]", "8", "INT2"]."""
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _fmt_node_params(node, params):
    fields = _CONFIG_TUPLE_FIELDS.get(node)
    bits = []
    for k, v in params.items():
        if k == "config" and fields:
            inner = v.strip()
            if inner.startswith("(") and inner.endswith(")"):
                inner = inner[1:-1]
            bits.extend(f"{name}={val}" for name, val in zip(fields, _split_top_level(inner)))
        else:
            bits.append(f"{k}={v}")
    return " ".join(bits)


def _format_feedback(node, records, max_cases=None):
    lines = []
    all_cases = tav_eval.expand_records(records)
    cases = all_cases if max_cases is None else all_cases[:max_cases]
    for c in cases:
        tag = tav_eval._case_tag(c)
        params = _fmt_node_params(node, c.get("params", {}))
        port = c.get("port")
        if port:
            rle_s = _fmt_rle(c["delta_vector"])
            lines.append(
                f"  [{tag}] {port} {params} | peak={c['peak_volume_delta']} "
                f"len_delta={c['len_delta']} delta=[{rle_s}]"
            )
        else:
            tail = (c.get("longrepr") or "").splitlines()
            lines.append(f"  [{tag}] {params} | {tail[-1] if tail else ''}")
    if max_cases is not None:
        more = len(all_cases) - max_cases
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
    model=None,
    previous_feedback=None,
    max_turns=30,
):
    """Return (passed, feedback_for_next_prompt, records, score).

    Splices workspace/get_tree_model.py into the node's source and runs its
    characterization pytest via tav_eval; this is the evaluator swapped into
    Bespoke's generate -> evaluate -> reprompt loop.

    If `model` is given, a second analyzer agent -- same model, its own
    fresh context, no file-writing tools used -- reviews the candidate
    source plus this iteration's (and the previous iteration's, if any)
    feedback, and its analysis is appended to the feedback returned here for
    the tree-generating agent's next prompt."""
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
    except (SystemExit, Exception) as e:
        # A malformed candidate (e.g. a bad patch leaving invalid Python
        # behind) can raise from deep inside tav_eval/replace_function
        # (SyntaxError, etc.) rather than the SystemExit it raises for
        # expected CLI-style errors -- catch both so one bad iteration
        # doesn't take down the whole run, and print the traceback (the
        # tee in run_loop captures it into the log) for debugging.
        print(f"tav_eval could not evaluate the candidate ({type(e).__name__}):\n{traceback.format_exc()}")
        return False, f"tav_eval could not evaluate the candidate ({type(e).__name__}): {e}", [], None

    sc = tav_eval.score_records(records)
    if sc["solved"]:
        return True, "", records, sc
    feedback = (
        f"score={round(sc['score'], 2)} (pass={sc['n_pass']} fail={sc['n_fail']} "
        f"error={sc['n_error']} skip={sc['n_skip']})\n" + _format_feedback(node, records)
    )

    if model is not None:
        analysis_task = build_analysis_task(
            node, candidate.read_text(), feedback, previous_feedback
        )
        analysis = run_agent(analysis_task, model, workspace / "analysis", max_turns=max_turns)
        feedback += f"\n\nAnalyzer feedback:\n{analysis}"

    return False, feedback, records, sc


# ── main loop ────────────────────────────────────────────────────────────────
def _baseline_path(node, src_override=None, test_override=None):
    entry = tav_eval.resolve_node(node, src_override, test_override)
    base = os.path.basename(entry["src"]).replace(".py", "_tree_model.py")
    return os.path.join(_TAV_EVAL_DIR, "examples", base)


DEFAULT_LOG_PATH = "llm_tree_modeling.log"


class _TeeStream:
    """Mirrors writes to the original stream and to a timestamped log file,
    so the log file captures everything printed during the loop -- our own
    progress lines, run_agent's verbose tool-call/result trace, and tracebacks
    -- without having to touch agent_stub's own print() calls."""

    def __init__(self, original, fh):
        self._original = original
        self._fh = fh
        self._buf = ""

    def write(self, s):
        self._original.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._fh.write(f"[{ts}] {line}\n")
        self._fh.flush()
        return len(s)

    def flush(self):
        self._original.flush()
        self._fh.flush()

    def isatty(self):
        return False


def run_loop(
    node,
    model=DEFAULT_MODEL,
    workspace=Path("workspace"),
    max_iterations=100,
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
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_fh)
    sys.stderr = _TeeStream(orig_stderr, log_fh)
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

        print(f"node={node} model={model} max_iterations={max_iterations} out_dir={out_dir}")

        previous, feedback = None, None
        for i in range(1, max_iterations + 1):
            print(f"\n{'=' * 60}\niteration {i}\n{'=' * 60}")
            task = build_task(node, src_path, baseline_src, previous=previous, feedback=feedback)
            run_agent(task, model, workspace, max_turns=max_turns)

            passed, feedback, records, sc = check_output(
                workspace, node,
                src_override=src_override, test_override=test_override,
                include_extra_tests=include_extra_tests, cache_dir=cache_dir,
                model=model, previous_feedback=feedback, max_turns=max_turns,
            )
            previous = candidate_path.read_text() if candidate_path.exists() else None

            if previous is not None:
                (cand_dir / f"iter_{i:03d}.py").write_text(previous)
            if sc is not None:
                print(f"\niter {i}: score={round(sc['score'], 2)} (pass={sc['n_pass']} fail={sc['n_fail']} "
                      f"error={sc['n_error']})")
                history.append({"iteration": i, **sc})
                if sc["score"] < best["score"] and previous is not None:
                    best = {"iteration": i, "source": previous, "score": sc["score"]}
            else:
                history.append({"iteration": i, "error": feedback})

            if passed:
                print(f"\nAll cases matched the rtlsim reference on iteration {i}.")
                break
            print(f"\nfeedback:\n{feedback}")
        else:
            print(f"\nStopped after {max_iterations} iterations.")

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
            print(f"applied best candidate (iteration {best['iteration']}) to {src_path}")

        print(f"\nbest candidate: {best_path}")
        print(f"history:        {out_dir / 'history.json'}")
        return best_path
    except Exception:
        print(traceback.format_exc())
        raise SystemExit(1)
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_fh.close()


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", help="FINN node name, e.g. ConvolutionInputGenerator (tav_eval.py --list)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--max-iterations", type=int, default=100)
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
