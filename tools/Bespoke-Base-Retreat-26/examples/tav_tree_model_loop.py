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
    You are an fpga expert at vitis hls and system verilog, you want to
    design characteristic tree models of ML operators described in this
    repository's deps/finn-hlslib and finn-rtllib directories. You are
    optimizing one node in this task, the node {node}. The source code of a
    tree that we already have is in {src_path} get_tree_model() function. The
    node's hls functionality is described in: {hls_ref} and rtl in {rtl_ref}.

    Each time you create your own tree model, we will execute it to produce a
    token access vector which we then compare vs an rtl-simulated ground
    truth. Your goal is to make your tree model produce an identical tav to
    rtlsim for a variety of testcases.

    The way token access vectors are produced using the tree is the following:
    We use a class called Characteristic_Node which is found in finn's src/finn/util/basic.py
    file. A Characterististic_Node encodes a list of states that the node is in
     (actual tree nodes) as well as the number of times the state is accessed repeatedly (values on an edge). 
     A leaf Characterististic_Node is a special case that encodes tuple which state if the current state is a write state,
    a read state, neither or both. Depending on this, in this exact state clock cycle, a produced TAV
    would have a +1 added to either the read or the write vector at that clock cycle.
    Characteristic_Nodes effectively attempt to provide a cycle-accurate model of the entire node, where
    the only property we want to really model is the state of the input and output channels (reads/writes).

    The plan for building a tree model is to to first extract all parameters of a node using self.get_nodeattr(...),
    which may affect what states the node will contain and how many times they will be accessed.
    These parameters are typically set at compile-time and so we wish to encode them in some way into the tree edges.

    You may look at tree modes of other nodes in the src/finn/custom_op folder for inspiration.
    Many of these trees are close to the rtl-sim equivalents in what token access vectors they produce,
    we typically have distinct trees for RTL and HLS-based nodes as their behavior may heavily warry.
    The primary design mistake that can be made is to wrongly assume when a read and a write overlap during a node's execution.

    You should first design a a tree that, when traversed, produces the input and output token access vectors that have correct volume:
    that is their total number of tokens reads and written matches rtlsim. Then you should start fusing phases or creating sub-phases such that
    you fuse reads and writes correctly (or introduce delay idle states) such that the length of the vector is also correct (how many cycles it took to execute).
    Lastly, you would make sure that the vectors are completely identical, this is the hard part where the fusing and partial states are most important.

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
):
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

    previous, feedback = None, None
    for i in range(1, max_iterations + 1):
        print(f"\n{'=' * 60}\niteration {i}\n{'=' * 60}")
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
            print(f"\niter {i}: score={sc['score']} (pass={sc['n_pass']} fail={sc['n_fail']} "
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
        cache_dir=args.cache_dir,
        apply_best=args.apply_best,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
