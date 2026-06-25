"""Iteration loop for FINN get_tree_model search, evaluated by tav_eval.

Same generate -> run -> evaluate -> reprompt shape as iteration_loop.py, split
across two agents with distinct roles:

- The tree-builder agent only ever writes get_tree_model.py. It never sees
  raw pytest/TAV output -- its prompt carries only the analyzer's distilled
  suggestions, its own previous attempt, and the node's reference source.
- The analyzer agent (run by check_output() in its own fresh context each
  iteration) is the only one exposed to the raw evaluation feedback. It
  reads the candidate plus that feedback and the node's source, and writes
  back concrete, actionable suggestions -- never editing the candidate
  itself. Those suggestions are what check_output() hands the builder next.

check_output() splices the candidate into the target FINN node (via
tools/tav_eval) and runs the node's analytical-characterization pytest in the
FINN docker container, comparing the analytical token access vector (TAV)
against the rtlsim reference. Stops when every case matches the reference
exactly, or after --max-iterations.

The TASK prompt below is a starting point -- tune it for whatever guidance
gets the model to converge faster; the evaluation side (check_output) doesn't
need to change when you do. Generic methodology shared by both agents (what a
tree model is, how to approach building one, the no-simulating-RTL rule)
lives in agent_stub.agent.SYSTEM_PROMPT, not here -- what's here is only the
node-specific facts (where its source is, which pytest runs it) and the
role-specific task for whichever agent is being prompted.

Run:
    python examples/tav_tree_model_loop.py ConvolutionInputGenerator
    python examples/tav_tree_model_loop.py FMPadding --model gpt-5.1 --max-iterations 10
    python examples/tav_tree_model_loop.py ConvolutionInputGenerator --apply-best
"""

from __future__ import annotations

import argparse
import ast
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

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_PACKAGE_DIR, "outputs")
INPUTS_DIR = os.path.join(_PACKAGE_DIR, "inputs")

CANDIDATE_FILENAME = "get_tree_model.py"
BEST_FILENAME = "best_get_tree_model.py"
ANALYZER_FEEDBACK_FILENAME = "analyzer_feedback.log"
RUN_SUMMARY_FILENAME = "run_summary.log"

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
            f"(unlisted node -- its backend file is not yet mirrored under "
            f"{INPUTS_DIR}; mirror it there before running this node)",
            "(unlisted node -- see above)",
        )
    return refs["hls"], refs["rtl"]


# ── prompt ───────────────────────────────────────────────────────────────────
# This is the part meant to be tuned by hand; check_output() below does not
# depend on its wording.
TASK_HEADER = textwrap.dedent("""\
    You are optimizing node {node}. The existing tree is in {src_path}'s
    get_tree_model() function. HLS reference: {hls_ref}. RTL reference:
    {rtl_ref}. Each iteration runs {test_ref} to score the candidate.

    You do not see that test's raw output directly -- a separate analyzer
    agent reviews it each iteration and reports back only its conclusions.
    What you get on a retry is the analyzer's suggestions, your own previous
    attempt, and the node's source -- never the raw pass/fail data itself.

    Three more files, not specific to {node}, are also cleaned and
    available under {inputs_dir}: hwcustomop.py
    ({inputs_dir}/src/finn/custom_op/fpgadataflow/hwcustomop.py) is the
    base class every node inherits, showing how FINN stores and exposes
    compile-time parameters via get_nodeattr; basic.py
    ({inputs_dir}/src/finn/util/basic.py) contains Characteristic_Node and
    the tree-traversal function that turns a tree into a TAV; test.py
    ({inputs_dir}/src/finn/util/test.py) contains the characterization
    pytest's shared helpers, useful for understanding what is actually
    being compared against rtlsim.

    Only read files under {inputs_dir} -- never read, grep, or list anything
    under the live FINN repository checked out at {finn_root}, for any
    reason (not the node's pytest file, not other transformation/analysis
    source, nothing). Every file you need -- the hls/rtl reference sources
    above, {src_path} itself, every other node's tree model under
    src/finn/custom_op/fpgadataflow, hwcustomop.py, and basic.py -- is
    already mirrored under {inputs_dir} at the same relative path (license/
    copyright headers stripped, nothing else changed), e.g.
    {inputs_dir}/deps/finn-hlslib/streamtools.h. Only writes are confined to
    your workspace.

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

    Feedback on that attempt:
    {feedback}

    Write an improved `{filename}` that addresses this feedback.
""")

# ── second agent: analysis only, no tree-writing ────────────────────────────
# Reviews the candidate plus this and the previous iteration's feedback in
# its own fresh context and proposes concrete structural fixes; its reply is
# appended to the feedback the tree-generating agent sees next (see
# check_output below), it never edits the candidate itself.
ANALYSIS_TASK = textwrap.dedent("""\
    You are reviewing a candidate characteristic tree model for node {node}.
    Its source is {src_path}. HLS reference: {hls_ref}. RTL reference:
    {rtl_ref}. Each iteration runs {test_ref} to score the candidate; its
    raw output (below) is given to you, and only to you.

    You are NOT generating or editing the tree yourself -- a separate
    tree-builder agent does that, and it is restricted to your conclusions
    plus its own previous tree and the node's source: it never sees the raw
    evaluation feedback below. Your job is to read the source, study the
    candidate below against its evaluation feedback, and write concrete,
    actionable suggestions for how the tree's structure should change (e.g.
    a missing state, a wrong repeat count, a misjudged read/write overlap, a
    phase that should be fused or split) to close the remaining gaps. Be
    specific -- the tree-builder agent only ever sees what you say here, not
    the data below. Reply with your analysis as plain text -- do not use
    apply_patch or write any files.

    Only read files under {inputs_dir} -- never read, grep, or list anything
    under the live FINN repository checked out at {finn_root}. Every file
    you need -- the hls/rtl reference sources above, {src_path} itself,
    every other node's tree model under src/finn/custom_op/fpgadataflow,
    hwcustomop.py, and basic.py -- is already mirrored under {inputs_dir} at
    the same relative path, e.g. {inputs_dir}/deps/finn-hlslib/streamtools.h.

    Three more files, not specific to {node}, are also available under
    {inputs_dir}: hwcustomop.py
    ({inputs_dir}/src/finn/custom_op/fpgadataflow/hwcustomop.py) is the
    base class every node inherits, showing how FINN stores and exposes
    compile-time parameters via get_nodeattr; basic.py
    ({inputs_dir}/src/finn/util/basic.py) contains Characteristic_Node and
    the tree-traversal function that turns a tree into a TAV; test.py
    ({inputs_dir}/src/finn/util/test.py) contains the characterization
    pytest's shared helpers, useful for understanding what is actually
    being compared against rtlsim.

    If the existing parametrize values on the characterization test (visible
    in the per-case `params=...` of the evaluation feedback below -- e.g.
    idim, pad, num_ch, simd, idt, depending on the node) are not enough to tell apart
    two competing theories about the node's behavior, you may propose
    adding ONE new value to ONE of those parameters by ending your reply
    with a line of exactly this form:
        PROPOSE_TEST_CASE: <param_name>=<python_literal>
    e.g. `PROPOSE_TEST_CASE: idim=[12, 8]` or `PROPOSE_TEST_CASE: simd=4`.
    The literal must be the same kind of value as the parameter's existing
    ones (e.g. a list for idim, an int for simd) and must not already be
    one of them. You may NOT propose this for `mode` or `impl_style` --
    those select which simulation/backend runs, not a property of the node,
    and any such proposal is rejected. You have no tool to edit the test
    file yourself; a line in this exact format is parsed by the harness
    running this conversation and, if valid, applied on your behalf before
    the next iteration -- at most one proposal is honored per reply, and
    most replies should have none at all. Only propose one when you have a
    specific, stated hypothesis the current cases can't distinguish.

    Keep proposed values small: each is a real rtlsim run, and must stay
    within roughly a few thousand cycles. As a rule of thumb, the product
    of the magnitudes in the value (e.g. idim=[H, W] is about H*W; simd=N
    is about N) should stay well under a few thousand -- e.g. idim=[20, 20]
    (~400) is fine, idim=[200, 200] (~40000) is not. An oversized proposal
    is rejected outright. If a proposal you made is reported back to you as
    rejected or rolled back, do not repeat the same value -- propose a
    smaller one or drop the idea.

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


def build_analysis_task(node, src_path, candidate_source, feedback, test_ref=None, previous_feedback=None):
    hls_ref, rtl_ref = _node_refs(node, src_path)
    previous_block = (
        PREVIOUS_FEEDBACK_BLOCK.format(previous_feedback=previous_feedback)
        if previous_feedback else ""
    )
    return ANALYSIS_TASK.format(
        node=node,
        src_path=src_path,
        hls_ref=hls_ref,
        rtl_ref=rtl_ref,
        test_ref=test_ref or "(not specified)",
        finn_root=tav_eval.FINN_ROOT,
        inputs_dir=INPUTS_DIR,
        filename=CANDIDATE_FILENAME,
        candidate=candidate_source.strip(),
        feedback=feedback,
        previous_block=previous_block,
    )


def build_task(node, src_path, baseline, test_ref=None, previous=None, feedback=None):
    hls_ref, rtl_ref = _node_refs(node, src_path)
    task = TASK_HEADER.format(
        node=node,
        src_path=src_path,
        hls_ref=hls_ref,
        rtl_ref=rtl_ref,
        test_ref=test_ref or "(not specified)",
        finn_root=tav_eval.FINN_ROOT,
        inputs_dir=INPUTS_DIR,
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


# Matches a `PROPOSE_TEST_CASE: <param>=<literal>` line in an analyzer
# reply (see ANALYSIS_TASK). Only the first match is ever honored.
_PROPOSE_TEST_CASE_RE = re.compile(r"^\s*PROPOSE_TEST_CASE:\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.MULTILINE)


def _apply_proposed_test_case(node, src_override, test_override, analysis):
    """Look for a `PROPOSE_TEST_CASE: <param>=<literal>` line in the
    analyzer's reply and, if present, apply the first one found by adding
    that one value to the node's characterization test via
    tav_eval.add_parametrize_value -- the analyzer never edits the test file
    itself (it has no file-writing tools); this is the harness doing it on
    its behalf, capped at one new value per analyzer call.

    Returns (pending_case, note):
    - pending_case is a dict describing the addition (for check_output to
      validate against the next real pytest run -- see
      _check_pending_test_case) if one was applied, else None.
    - note is a human-readable explanation to fold into the feedback when
      the proposal was rejected up front (protected param, duplicate,
      oversized value, etc. -- see add_parametrize_value), else None. A
      proposal that gets applied here but later turns out to break pytest
      is instead reported by _check_pending_test_case next iteration."""
    matches = _PROPOSE_TEST_CASE_RE.findall(analysis)
    if not matches:
        return None, None
    if len(matches) > 1:
        print(f"[orchestrator] analyzer proposed {len(matches)} new test cases; "
              "only the first is honored this iteration.")
    param_name, value_literal = matches[0]

    entry = tav_eval.resolve_node(node, src_override, test_override)
    test_file, _, func_name = entry["test"].partition("::")
    if not os.path.isabs(test_file):
        test_file = os.path.join(tav_eval.FINN_ROOT, test_file)

    try:
        tav_eval.add_parametrize_value(test_file, func_name, param_name, value_literal)
    except SystemExit as e:
        note = (
            f"[orchestrator] Rejected proposed test case {param_name}={value_literal}: {e}. "
            "Propose a different value (or none) next time."
        )
        print(note)
        return None, note

    value_repr = repr(ast.literal_eval(value_literal))
    print(f"[orchestrator] added new test case: {param_name}={value_literal} (added to {entry['test']})")
    pending_case = {
        "test_file": test_file,
        "func_name": func_name,
        "test_nodeid": entry["test"],
        "param_name": param_name,
        "value_literal": value_literal,
        "value_repr": value_repr,
    }
    return pending_case, None


def _check_pending_test_case(pending_case, records):
    """Validate a test case added on the *previous* call's analyzer turn now
    that a real pytest run (the one that just produced ``records``) has
    actually exercised it -- this is the rollback half of the safety net:
    a value that's syntactically a valid literal can still make pytest fail
    at collection (e.g. a shape pytest's own setup can't handle) or crash a
    specific case at execution time (an ERROR record, not a TAV mismatch).
    Either way it gets rolled back via tav_eval.remove_parametrize_value --
    precise enough to undo just this one addition, not any earlier-accepted
    ones -- and is never proposed again automatically.

    Returns (records, note):
    - records is unchanged if the case is fine, or has any record(s) for the
      rolled-back value filtered out otherwise (an empty ``records`` --
      collection failed outright -- is returned as-is; the caller must
      re-run evaluate_tree_model to get something to score/report on).
    - note is None if the case is fine, else a human-readable explanation of
      the rollback for the next prompt's feedback."""
    if pending_case is None:
        return records, None

    pname, vrepr = pending_case["param_name"], pending_case["value_repr"]
    collection_failed = not records
    matches = [r for r in records if r.get("params", {}).get(pname) == vrepr]
    errored = any(tav_eval._verdict_tag(r) == "ERROR" for r in matches)
    # the plugin only writes a record for a test item that reached its "call"
    # phase (see _tav_eval_plugin.py) -- a crash during pytest's own *setup*
    # phase (e.g. a fixture choking on the new value) leaves no record for
    # this value at all, silently, even though sibling records exist for
    # everything else. Treat that as a failure too rather than missing it.
    no_evidence = not matches and bool(records)

    if not (collection_failed or errored or no_evidence):
        return records, None  # the value survived a real run -- keep it

    try:
        tav_eval.remove_parametrize_value(
            pending_case["test_file"], pending_case["func_name"], pname, pending_case["value_literal"]
        )
    except SystemExit as e:
        # shouldn't normally happen (we just added it last iteration), but
        # don't let a rollback failure crash the loop
        print(f"[orchestrator] WARNING: could not roll back {pname}={vrepr}: {e}")
        return records, None

    if collection_failed:
        reason = "broke pytest collection entirely (treated as an invalid/malformed value)"
    elif errored:
        reason = "crashed during execution (an error, not a TAV mismatch)"
    else:
        reason = "produced no result at all (likely crashed during pytest setup, before any comparison ran)"
    note = (
        f"[orchestrator] The test case you proposed last iteration, "
        f"{pname}={pending_case['value_literal']}, {reason} and has been rolled back out of "
        f"{pending_case['test_nodeid']}. Do not propose this exact value again."
    )
    print(note)
    if collection_failed:
        return records, note  # caller must re-evaluate; nothing to filter
    return [r for r in records if r not in matches], note


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
    iteration=None,
    prompts_log_fh=None,
    pending_case=None,
):
    """Return (passed, eval_feedback, builder_feedback, records, score, pending_case).

    Splices workspace/get_tree_model.py into the node's source and runs its
    characterization pytest via tav_eval; this is the evaluator swapped into
    Bespoke's generate -> evaluate -> reprompt loop.

    The two feedback strings enforce the builder/analyzer split: `eval_feedback`
    is the raw score plus per-case deltas (plus any orchestrator rollback/
    test-case notes) -- meant for logs and as the *next* analyzer call's
    `previous_feedback`, never for the tree-builder. `builder_feedback` is
    what actually goes in the tree-builder's next prompt: the analyzer's
    distilled reply when `model` is given (it defaults to `eval_feedback`
    only when there's no analyzer call to produce something better, i.e.
    `model is None` or an early error/solved exit).

    `pending_case`, if given, describes a parametrize value the analyzer got
    added to the test file on the *previous* call (see
    _apply_proposed_test_case) -- this call's real pytest run is the first
    one to actually exercise it, so it's checked here (see
    _check_pending_test_case) and rolled back with feedback explaining why
    if it broke pytest collection or crashed during execution; a value that
    survives is never re-checked. The returned `pending_case` describes
    whatever the analyzer proposes *this* call, for the next call to check
    in turn.

    If `model` is given, a second analyzer agent -- same model, its own
    fresh context, no file-writing tools used -- reviews the candidate
    source plus this iteration's (and the previous iteration's, if any) raw
    evaluation feedback, and its analysis becomes `builder_feedback` for the
    tree-generating agent's next prompt. The analyzer may also propose
    widening the characterization test's coverage by one parametrize value
    (see _apply_proposed_test_case); if accepted (or rejected, or rolled
    back), that's noted in `eval_feedback` (and, since it's part of the
    analyzer's own reply, also reaches the builder via `builder_feedback`)."""
    candidate = workspace / CANDIDATE_FILENAME
    if not candidate.exists():
        msg = f"{CANDIDATE_FILENAME} was not created in the workspace."
        return False, msg, msg, [], None, pending_case

    def _evaluate():
        return tav_eval.evaluate_tree_model(
            node,
            str(candidate),
            src_override=src_override,
            test_override=test_override,
            include_extra_tests=include_extra_tests,
            cache_dir=cache_dir,
            quiet=True,
            return_records=True,
        )

    try:
        _, records = _evaluate()
    except (SystemExit, Exception) as e:
        # A malformed candidate (e.g. a bad patch leaving invalid Python
        # behind) can raise from deep inside tav_eval/replace_function
        # (SyntaxError, etc.) rather than the SystemExit it raises for
        # expected CLI-style errors -- catch both so one bad iteration
        # doesn't take down the whole run, and print the traceback (the
        # tee in run_loop captures it into the log) for debugging.
        print(f"tav_eval could not evaluate the candidate ({type(e).__name__}):\n{traceback.format_exc()}")
        msg = f"tav_eval could not evaluate the candidate ({type(e).__name__}): {e}"
        return False, msg, msg, [], None, pending_case

    rollback_note = None
    if pending_case is not None:
        records, rollback_note = _check_pending_test_case(pending_case, records)
        pending_case = None  # resolved either way -- good or rolled back, don't recheck it
        if rollback_note is not None and not records:
            # collection failed outright -- the rollback just fixed the test
            # file, but `records` is empty (nothing ran), so re-evaluate to
            # get something real to score and report this iteration.
            try:
                _, records = _evaluate()
            except (SystemExit, Exception) as e:
                print(f"tav_eval could not re-evaluate after rollback ({type(e).__name__}):\n"
                      f"{traceback.format_exc()}")
                msg = (f"{rollback_note}\n\ntav_eval could not re-evaluate after rollback "
                       f"({type(e).__name__}): {e}")
                return False, msg, msg, [], None, None

    sc = tav_eval.score_records(records)
    if sc["solved"]:
        return True, "", "", records, sc, pending_case
    eval_feedback = (
        f"score={round(sc['score'], 2)} (pass={sc['n_pass']} fail={sc['n_fail']} "
        f"error={sc['n_error']} skip={sc['n_skip']})\n" + _format_feedback(node, records)
    )
    if rollback_note is not None:
        eval_feedback = f"{rollback_note}\n\n{eval_feedback}"
    builder_feedback = eval_feedback  # fallback when there's no analyzer to distill it

    if model is not None:
        entry = tav_eval.resolve_node(node, src_override, test_override)
        analysis_task = build_analysis_task(
            node, entry["src"], candidate.read_text(), eval_feedback,
            test_ref=entry["test"], previous_feedback=previous_feedback,
        )
        if prompts_log_fh is not None:
            _log_prompt(prompts_log_fh, iteration, "analyzer", analysis_task)
        analysis = run_agent(analysis_task, model, workspace / "analysis", max_turns=max_turns)
        builder_feedback = analysis

        new_pending, note = _apply_proposed_test_case(node, src_override, test_override, analysis)
        if new_pending is not None:
            eval_feedback += (
                f"\n\n[orchestrator] Added new test case to the validator: "
                f"{new_pending['param_name']}={new_pending['value_literal']} "
                f"(added to {new_pending['test_nodeid']}). This will be exercised -- and checked "
                "for validity -- starting next iteration."
            )
            pending_case = new_pending
        elif note is not None:
            eval_feedback += f"\n\n{note}"

    return False, eval_feedback, builder_feedback, records, sc, pending_case


# ── main loop ────────────────────────────────────────────────────────────────
def _baseline_path(node, src_override=None, test_override=None):
    entry = tav_eval.resolve_node(node, src_override, test_override)
    base = os.path.basename(entry["src"]).replace(".py", "_tree_model.py")
    return os.path.join(_TAV_EVAL_DIR, "examples", base)


def _default_log_path():
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    return os.path.join(OUTPUTS_DIR, f"tree-model-run-{ts}-output.log")


def _prompts_log_path(log_path):
    """Same run, sibling file: <name-without-.log>_prompts.log."""
    return log_path.with_name(log_path.stem + "_prompts.log")


PROMPT_LOG_SEP = "-" * 59


def _log_prompt(fh, iteration, role, prompt):
    """Record the exact prompt sent to an agent this iteration, so the run can
    be audited without re-deriving prompts from the (possibly since-changed)
    templates above."""
    fh.write(f"{PROMPT_LOG_SEP}\n")
    fh.write(f"iteration: {iteration}, {role} prompt: {prompt}\n")
    fh.flush()


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


def _safe_node_name(node):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", node)


def _node_outputs_dir(node):
    """Durable, run-independent home for this node's best result -- unlike
    out_dir, which is a fresh timestamped directory every run."""
    return Path(OUTPUTS_DIR) / _safe_node_name(node)


def _persist_best_for_node(node, source, feedback, iteration, score):
    """Overwrite outputs/<node>/'s best candidate and the analyzer feedback
    that earned it. Callers only invoke this at a genuine new-best update (a
    validly-scored baseline or a strictly-improving iteration), so the most
    recent write here is always the best one found across the whole run --
    including a final, all-passing iteration, since this is called from the
    same spot that updates the in-memory `best` dict, before the loop's `if
    passed: break`."""
    node_dir = _node_outputs_dir(node)
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / BEST_FILENAME).write_text(source)
    if not feedback:
        # check_output() returns empty feedback only when this candidate
        # solved every case outright -- there was nothing left to analyze,
        # not a missing/failed analyzer call. Say so explicitly rather than
        # leaving the log looking broken.
        feedback = (
            f"(no analyzer feedback for iteration {iteration} -- this candidate "
            "passed every case, so it was never sent to the analyzer.)"
        )
    (node_dir / ANALYZER_FEEDBACK_FILENAME).write_text(
        f"iteration {iteration} (score={round(score, 4)}):\n{feedback}\n"
    )


def _write_run_summary_for_node(node, elapsed_minutes, iterations_run, best_record):
    """Append a one-line report to outputs/<node>/run_summary.log once a
    node's whole run_loop() (baseline + every iteration) has finished,
    however it finished -- early pass or max_iterations exhaustion."""
    if best_record is not None and "n_pass" in best_record:
        score_str = f"{best_record['n_pass']}/{best_record['n_total']} tests passed"
    else:
        score_str = "N/A (no iteration produced a valid evaluation)"
    node_dir = _node_outputs_dir(node)
    node_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"[{timestamp}] time={elapsed_minutes:.2f}min iterations={iterations_run} "
        f"score={score_str}\n"
    )
    with (node_dir / RUN_SUMMARY_FILENAME).open("a") as fh:
        fh.write(line)


def run_loop(
    node,
    model=DEFAULT_MODEL,
    workspace=Path("workspace"),
    max_iterations=3,
    max_turns=30,
    baseline=None,
    src_override=None,
    test_override=None,
    include_extra_tests=False,
    cache_dir=None,
    apply_best=False,
    out_dir=None,
    log_path=None,
):
    log_path = Path(log_path) if log_path is not None else Path(_default_log_path())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", buffering=1)
    prompts_fh = _prompts_log_path(log_path).open("w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_fh)
    sys.stderr = _TeeStream(orig_stderr, log_fh)
    run_start = datetime.datetime.now()
    try:
        entry = tav_eval.resolve_node(node, src_override, test_override)
        src_path = entry["src"]
        test_ref = entry["test"]

        baseline = baseline or _baseline_path(node, src_override, test_override)
        if not os.path.isfile(baseline):
            raise SystemExit(f"baseline candidate not found: {baseline}")

        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_node = _safe_node_name(node)
        out_dir = Path(out_dir or os.path.join(
            tav_eval._host_build_dir(), "tav_bespoke", f"{safe_node}-{run_id}"
        ))
        cand_dir = out_dir / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)

        workspace.mkdir(parents=True, exist_ok=True)
        candidate_path = workspace / CANDIDATE_FILENAME

        baseline_src = Path(baseline).read_text()
        history = []
        iterations_run = 0

        print(f"node={node} model={model} max_iterations={max_iterations} out_dir={out_dir}")

        # Baseline pass: run the node's existing tree model through the
        # validator before any LLM involvement, send that result to the
        # analyzer, and seed iteration 1's prompt with both -- so the first
        # tree-builder prompt already carries real feedback (and the
        # analyzer's read on it) instead of flying blind.
        print(f"\n{'=' * 60}\nbaseline (node's existing tree model)\n{'=' * 60}")
        candidate_path.write_text(baseline_src)
        passed0, eval_feedback0, builder_feedback0, records0, sc0, pending_case = check_output(
            workspace, node,
            src_override=src_override, test_override=test_override,
            include_extra_tests=include_extra_tests, cache_dir=cache_dir,
            model=model, previous_feedback=None, max_turns=max_turns,
            iteration=0, prompts_log_fh=prompts_fh, pending_case=None,
        )
        if sc0 is not None:
            print(f"\nbaseline: score={round(sc0['score'], 2)} (pass={sc0['n_pass']} fail={sc0['n_fail']} "
                  f"error={sc0['n_error']})")
            history.append({"iteration": 0, **sc0})
            # seed with the baseline's real score (not inf) -- it can legitimately
            # win against a worse LLM iteration; a passing baseline short-circuits
            # below and never reaches this comparison anyway.
            best = {"iteration": 0, "source": baseline_src, "score": sc0["score"]}
            _persist_best_for_node(node, baseline_src, builder_feedback0, 0, sc0["score"])
        else:
            history.append({"iteration": 0, "error": eval_feedback0})
            # tav_eval couldn't even evaluate the baseline -- no real score to
            # seed with, so any successful later iteration should still win.
            best = {"iteration": 0, "source": baseline_src, "score": float("inf")}

        if passed0:
            print("\nThe node's existing tree model already matches the rtlsim reference; nothing to do.")
            best = {"iteration": 0, "source": baseline_src, "score": 0.0}
        else:
            print(f"\nbaseline eval feedback:\n{eval_feedback0}")
            if builder_feedback0 != eval_feedback0:
                print(f"\nbaseline analyzer feedback (goes to the builder):\n{builder_feedback0}")

            previous, builder_feedback, analyzer_prev_feedback = baseline_src, builder_feedback0, eval_feedback0
            for i in range(1, max_iterations + 1):
                iterations_run = i
                print(f"\n{'=' * 60}\niteration {i}\n{'=' * 60}")
                task = build_task(
                    node, src_path, baseline_src, test_ref=test_ref,
                    previous=previous, feedback=builder_feedback,
                )
                _log_prompt(prompts_fh, i, "tree-builder", task)
                run_agent(task, model, workspace, max_turns=max_turns)

                passed, eval_feedback, builder_feedback, records, sc, pending_case = check_output(
                    workspace, node,
                    src_override=src_override, test_override=test_override,
                    include_extra_tests=include_extra_tests, cache_dir=cache_dir,
                    model=model, previous_feedback=analyzer_prev_feedback, max_turns=max_turns,
                    iteration=i, prompts_log_fh=prompts_fh, pending_case=pending_case,
                )
                analyzer_prev_feedback = eval_feedback
                previous = candidate_path.read_text() if candidate_path.exists() else None

                if previous is not None:
                    (cand_dir / f"iter_{i:03d}.py").write_text(previous)
                if sc is not None:
                    print(f"\niter {i}: score={round(sc['score'], 2)} (pass={sc['n_pass']} fail={sc['n_fail']} "
                          f"error={sc['n_error']})")
                    history.append({"iteration": i, **sc})
                    if sc["score"] < best["score"] and previous is not None:
                        best = {"iteration": i, "source": previous, "score": sc["score"]}
                        _persist_best_for_node(node, previous, builder_feedback, i, sc["score"])
                else:
                    history.append({"iteration": i, "error": eval_feedback})

                if passed:
                    print(f"\nAll cases matched the rtlsim reference on iteration {i}.")
                    break
                print(f"\neval feedback:\n{eval_feedback}")
                if builder_feedback != eval_feedback:
                    print(f"\nanalyzer feedback (goes to the builder):\n{builder_feedback}")
            else:
                print(f"\nStopped after {max_iterations} iterations.")

        elapsed_minutes = (datetime.datetime.now() - run_start).total_seconds() / 60.0
        best_record = history[best["iteration"]] if best["iteration"] < len(history) else None
        _write_run_summary_for_node(node, elapsed_minutes, iterations_run, best_record)

        best_path = out_dir / BEST_FILENAME
        best_path.write_text(best["source"])
        (out_dir / "history.json").write_text(json.dumps({"node": node, "best": best["iteration"],
                                                            "best_score": best["score"],
                                                            "history": history}, indent=1))

        if not os.path.isabs(src_path):
            src_path = os.path.join(tav_eval.FINN_ROOT, src_path)
        tav_eval.restore_original(src_path)

        test_file = entry["test"].partition("::")[0]
        if not os.path.isabs(test_file):
            test_file = os.path.join(tav_eval.FINN_ROOT, test_file)
        if tav_eval.restore_original(test_file):
            print(f"restored {test_file} (reverted any analyzer-proposed test cases)")

        if apply_best:
            tav_eval.replace_function(src_path, str(best_path))
            print(f"applied best candidate (iteration {best['iteration']}) to {src_path}")

        print(f"\nbest candidate: {best_path}")
        print(f"history:        {out_dir / 'history.json'}")
        print(f"durable copy for {node}: {_node_outputs_dir(node)}")
        print(f"run summary:    {_node_outputs_dir(node) / RUN_SUMMARY_FILENAME} "
              f"(time={elapsed_minutes:.2f}min iterations={iterations_run})")
        return best_path
    except Exception:
        print(traceback.format_exc())
        raise SystemExit(1)
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_fh.close()
        prompts_fh.close()


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
    ap.add_argument("--log", default=None,
                     help="log file to tail while the loop runs (default: "
                          f"{OUTPUTS_DIR}/tree-model-run-<timestamp>-output.log)")
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
