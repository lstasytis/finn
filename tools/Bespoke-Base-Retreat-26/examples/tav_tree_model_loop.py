"""AlphaEvolve-style optimizer for FINN ``get_tree_model`` functions, rebuilt to
maximize solve rate (cost/turns are explicitly not a concern).

What changed vs. the original blind loop
----------------------------------------
The original wrote a file blind, waited a full dockerized pytest, and showed the
agent only a *delta* against its own guess -- one information-starved evaluation
per iteration, with no memory across iterations. This version:

1. **Local oracle + visible target.** The agent calls an ``eval_tree_model`` tool
   that scores its candidate *in-process* against the cached rtlsim reference and
   shows it the TARGET vector, its vector, and the delta -- as often as it wants.
   The local oracle (agent_stub/tav_runtime.py) is a faithful copy of FINN's
   analytical derivation, self-tested on the baseline so a divergence is caught
   before it can mislead. Backend selectable via ``--oracle {local,docker,both}``.
   Docker remains the source of truth: a locally-solved candidate is **confirmed**
   with a real docker run before it counts.

2. **Population / portfolio search** (``--parallel N``): N agents per iteration at
   spread temperatures, all candidates pooled into an archive, global best kept.

3. **Archive + recombination** (``--recombine K``): trees that each solve part of
   the problem are fed back with an explicit "combine these" instruction.

4. **Curriculum** (``--curriculum``): start on the smallest test case, lock it,
   then widen one case at a time -- far easier than fitting everything at once.

5. **Persistent memory** (``--memory``): the best lineage's conversation carries
   forward across iterations instead of restarting cold each time.

6. **Domain-expert system prompt + strongest model + high reasoning effort** by
   default (see agent_stub/agent.py:TAV_SYSTEM_PROMPT and models.py).

Run:
    python examples/tav_tree_model_loop.py ConvolutionInputGenerator
    python examples/tav_tree_model_loop.py FMPadding --model gpt-5.1 --parallel 8 --curriculum
    python examples/tav_tree_model_loop.py MVAU --oracle both --memory --apply-best
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import textwrap
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from agent_stub import progress
from agent_stub.agent import TAV_SYSTEM_PROMPT, run_agent_with_history
from agent_stub.models import DEFAULT_MODEL, get_model
from agent_stub.oracle import Oracle
from agent_stub.tools.bash import run_bash
from agent_stub.tools.eval_tree import EVAL_TOOL_SCHEMA, make_eval_handler
from agent_stub.tools.run import run_program

# tav_eval lives one level up in the FINN repo (tools/tav_eval).
_TAV_EVAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tav_eval")
)
if _TAV_EVAL_DIR not in sys.path:
    sys.path.insert(0, _TAV_EVAL_DIR)
import tav_eval  # noqa: E402

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(_PACKAGE_DIR, "outputs")
# per-agent transcripts (one file per worker per iteration, plus the analyzer):
# prompts, tool calls, tool feedback and the final reply for each agent run.
AGENTS_DIR = os.path.join(OUTPUTS_DIR, "agents")
INPUTS_DIR = os.path.join(_PACKAGE_DIR, "inputs")

CANDIDATE_FILENAME = "get_tree_model.py"

# ── per-node HLS/RTL reference pointers (prompt content) ────────────────────
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


def _node_refs(node):
    refs = NODE_REFS.get(node)
    if refs is None:
        return (
            f"(unlisted node -- mirror its backend under {INPUTS_DIR} first)",
            "(unlisted node -- see above)",
        )
    return refs["hls"], refs["rtl"]


# ── prompts (hand-tunable; the eval side does not depend on wording) ────────
TASK_TEMPLATE = textwrap.dedent("""\
    Node under optimization: {node} (FINN fpgadataflow).
    HLS reference: {hls_ref}
    RTL reference: {rtl_ref}
    Reference source is mirrored read-only under {inputs_dir} (same relative
    paths, headers stripped); read it with bash. Never read the live FINN repo.

    Your goal: write `get_tree_model(self)` so that, for EVERY active test case,
    both the input and output token access vectors match the rtlsim reference
    EXACTLY (zero delta). Use the `eval_tree_model` tool to see the target and
    your delta -- call it as many times as you need.

    Active test cases this round: {n_active}{curriculum_note}

    Recommended approach:
      1. Call eval_tree_model with the baseline below to see each port's TARGET.
      2. Describe the target's structure (period, run-lengths of read/write
         steps, idle gaps) and map it to the loop nest in the reference source.
      3. Encode it as a tree, eval, and drive the delta to zero -- one port and
         one case at a time, then make a single tree that covers all of them.

    Baseline get_tree_model (your starting point):
    ```python
    {baseline}
    ```
    {extra}
    When every active port reads 'exact', stop and reply with a one-line summary.
""")

RETRY_BLOCK = textwrap.dedent("""
    Best candidate so far (score {score}):
    ```python
    {best}
    ```
    Its latest evaluation:
    {feedback}
    """)

RECOMBINE_BLOCK = textwrap.dedent("""
    These archived trees each solve PART of the problem. Study what structure
    each gets right and synthesize a single tree that satisfies all cases:
    {partials}
    """)

ANALYZER_TEMPLATE = textwrap.dedent("""\
    You are an FPGA expert reviewing a candidate characteristic tree for node
    {node}. You do NOT write trees -- another agent does. Read the candidate and
    its per-port TARGET/YOURS/DELTA feedback and write concrete, specific
    structural fixes (a missing state, a wrong repeat count, a misjudged
    read/write overlap, a phase to fuse or split). Reference source is under
    {inputs_dir} (read-only, via bash); never read the live FINN repo. Reply with
    plain-text analysis only -- do not write files.

    HLS reference: {hls_ref}
    RTL reference: {rtl_ref}

    Candidate:
    ```python
    {candidate}
    ```

    Evaluation feedback:
    {feedback}
""")


# ── logging plumbing (kept from the original) ───────────────────────────────
class _TeeStream:
    def __init__(self, original, fh):
        self._original = original
        self._fh = fh
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
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


def _default_log_path():
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
    return os.path.join(OUTPUTS_DIR, f"tree-model-run-{ts}-output.log")


# ── archive ────────────────────────────────────────────────────────────────
class Archive:
    """Thread-safe pool of every candidate tried, plus the best tree found for
    each distinct subset of (case, port) pairs it gets exactly right -- the raw
    material for recombination."""

    def __init__(self):
        self._lock = threading.Lock()
        self.all = []  # (source, eval_result)
        self.best_for_subset = {}  # frozenset((tag, port)) -> (n_matched, source)

    def record(self, source, result):
        with self._lock:
            self.all.append((source, result))
            matched = frozenset(
                (c.tag, p.port)
                for c in result.cases
                for p in getattr(c, "ports", [])
                if p.matched
            )
            if matched:
                cur = self.best_for_subset.get(matched)
                if cur is None or len(source) < len(cur[1]):
                    self.best_for_subset[matched] = (len(matched), source)

    def partials(self, k):
        """Up to k diverse partial solutions (largest distinct matched-subsets)."""
        with self._lock:
            items = sorted(self.best_for_subset.items(), key=lambda kv: -len(kv[0]))
            return [src for _subset, (_n, src) in items[:k]]


def _record_callback(archive):
    def cb(source, result):
        archive.record(source, result)
    return cb


# ── one agent worker ────────────────────────────────────────────────────────
def _make_budgeted_reader(budget):
    """Wrap bash/run with a shared per-worker call budget. After ``budget``
    source-reading calls, further bash/run return a nudge to iterate via
    eval_tree_model instead -- this is the structural cure for the agent burning
    its whole turn budget paging through RTL source instead of testing trees.
    apply_patch and eval_tree_model are never budgeted."""
    state = {"n": 0}

    def bash_h(args, ws):
        state["n"] += 1
        if budget and state["n"] > budget:
            return (f"[source-reading budget of {budget} calls is spent for this attempt. "
                    "Stop reading source now and ITERATE on your tree with eval_tree_model -- "
                    "the target vectors in the feedback are the ground truth, not the RTL.]")
        return run_bash(args.get("command", ""), ws)

    def run_h(args, ws):
        state["n"] += 1
        if budget and state["n"] > budget:
            return (f"[source-reading budget of {budget} calls is spent. Iterate with "
                    "eval_tree_model now instead of running more analysis scripts.]")
        return run_program(args.get("path", ""), ws, args.get("args"))

    return bash_h, run_h


def _run_worker(worker_id, task, model_obj, workspace, max_turns, oracle, archive,
                seed_messages, bash_budget=25, transcript_path=None):
    """Run a single agent with the eval tool bound. Returns its message history
    (for memory) and its last candidate source (if any)."""
    ws = workspace / f"pop{worker_id}"
    ws.mkdir(parents=True, exist_ok=True)
    handler = make_eval_handler(oracle, on_result=_record_callback(archive))
    bash_h, run_h = _make_budgeted_reader(bash_budget)
    tool_handlers = {"eval_tree_model": handler, "bash": bash_h, "run": run_h}
    extra_tools = [EVAL_TOOL_SCHEMA]

    if seed_messages:
        messages = [dict(m) for m in seed_messages] + [{"role": "user", "content": task}]
    else:
        messages = [
            {"role": "system", "content": TAV_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
    try:
        _result, msgs = run_agent_with_history(
            messages, model_obj, ws, max_turns=max_turns,
            extra_tools=extra_tools, tool_handlers=tool_handlers,
            transcript_path=transcript_path,
        )
    except Exception:
        print(f"[worker {worker_id}] crashed:\n{traceback.format_exc()}")
        return messages, (ws / CANDIDATE_FILENAME).read_text() if (ws / CANDIDATE_FILENAME).exists() else None
    cand = (ws / CANDIDATE_FILENAME).read_text() if (ws / CANDIDATE_FILENAME).exists() else None
    return msgs, cand


def _temperature_portfolio(base_model, n):
    """Spread temperatures across the population for exploration diversity.

    Reasoning models (gpt-5.x with reasoning_effort, or Responses-API models)
    reject a non-default temperature, so we leave them untouched -- their
    population diversity comes from independent stochastic samples instead.
    """
    if n == 1:
        return [base_model]
    if getattr(base_model, "reasoning_effort", None) or base_model.use_responses:
        return [base_model] * n
    lo, hi = 0.4, 1.1
    return [replace(base_model, temperature=round(lo + (hi - lo) * i / (n - 1), 2))
            for i in range(n)]


# ── main loop ────────────────────────────────────────────────────────────────
def _baseline_path(node, src_override=None, test_override=None):
    entry = tav_eval.resolve_node(node, src_override, test_override)
    base = os.path.basename(entry["src"]).replace(".py", "_tree_model.py")
    return os.path.join(_TAV_EVAL_DIR, "examples", base)


def run_loop(
    node,
    model=DEFAULT_MODEL,
    workspace=Path("workspace"),
    max_iterations=20,
    max_turns=60,
    parallel=4,
    bash_budget=25,
    oracle_mode="local",
    curriculum=False,
    memory=False,
    analyzer=False,
    recombine=0,
    docker_confirm=True,
    reasoning_effort=None,
    feedback_cap=120,
    feedback_budget=40000,
    feedback_detail_cases=10,
    baseline=None,
    src_override=None,
    test_override=None,
    include_extra_tests=False,
    cache_dir=None,
    apply_best=False,
    out_dir=None,
    log_path=None,
    progress_table=None,
):
    t_start = time.monotonic()
    log_path = Path(log_path) if log_path is not None else Path(_default_log_path())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(orig_stdout, log_fh)
    sys.stderr = _TeeStream(orig_stderr, log_fh)
    try:
        base_model = get_model(model)
        if reasoning_effort is not None:
            base_model = replace(base_model, reasoning_effort=reasoning_effort or None)

        entry = tav_eval.resolve_node(node, src_override, test_override)
        src_path = entry["src"]
        baseline = baseline or _baseline_path(node, src_override, test_override)
        if not os.path.isfile(baseline):
            raise SystemExit(f"baseline candidate not found: {baseline}")
        baseline_src = Path(baseline).read_text()

        hls_ref, rtl_ref = _node_refs(node)

        run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_node = re.sub(r"[^A-Za-z0-9_.-]", "_", node)
        out_dir = Path(out_dir or os.path.join(
            tav_eval._host_build_dir(), "tav_bespoke", f"{safe_node}-{run_id}"))
        cand_dir = out_dir / "candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)

        print(f"node={node} model={base_model.name} oracle={oracle_mode} parallel={parallel} "
              f"curriculum={curriculum} memory={memory} max_iterations={max_iterations}")
        print(f"out_dir={out_dir}")

        oracle = Oracle(
            node, mode=oracle_mode, src_override=src_override, test_override=test_override,
            cache_dir=cache_dir, include_extra_tests=include_extra_tests,
            target_display_cap=feedback_cap,
            feedback_char_budget=feedback_budget,
            max_detail_cases=feedback_detail_cases,
        )

        # ── baseline docker run: populate reference vectors + node metadata ──
        print(f"\n{'=' * 60}\nbaseline docker run (capturing reference TAVs)\n{'=' * 60}")
        workspace_baseline = workspace / "baseline"
        workspace_baseline.mkdir(parents=True, exist_ok=True)
        (workspace_baseline / CANDIDATE_FILENAME).write_text(baseline_src)
        d0 = oracle.evaluate_docker(str(workspace_baseline / CANDIDATE_FILENAME))
        oracle.load_cases(d0["records"])
        if not oracle.cases:
            n_rec = len(d0["records"])
            n_empty_ref = sum(
                1 for r in d0["records"]
                if not any(p.get("rtlsim_vector") for p in r.get("ports", []))
            )
            if n_rec and n_empty_ref == n_rec:
                raise SystemExit(
                    f"baseline ran {n_rec} case(s) but EVERY one has an empty rtlsim "
                    "reference vector -- i.e. an rtlsim cache MISS (and no simulator to "
                    "generate it). The cached references are test-specific: e.g. the "
                    "committed cache is for the downsampler test, so run\n"
                    "  --test tests/fpgadataflow/test_fpgadataflow_downsampler.py::"
                    "test_fpgadataflow_analytical_characterization_downsampler\n"
                    "or populate the cache for this node's test first (see tav_eval README)."
                )
            raise SystemExit(
                "no reference cases captured from the baseline docker run -- "
                "check the rtlsim cache (see tav_eval README) and that the plugin loaded."
            )
        print(f"captured {len(oracle.cases)} case(s); baseline docker score={round(d0['score'], 2)} "
              f"solved={d0['solved']}")

        # ── prove the local oracle is faithful before trusting it ──
        if oracle_mode != "docker":
            ok, msg = oracle.selftest(baseline_src)
            print(f"[oracle self-test] {msg}")
            if not ok:
                if docker_confirm:
                    print("[oracle self-test] WARNING: local oracle diverges from FINN; "
                          "docker confirmation will still gate 'solved', but local feedback may "
                          "be unreliable. Consider --oracle docker for this node.")
                else:
                    raise SystemExit(
                        "local oracle self-test failed and --no-docker-confirm is set; refusing "
                        "to run with unverified feedback. Use --oracle docker.")

        if d0["solved"]:
            print("\nBaseline already matches rtlsim exactly; nothing to do.")
            progress.record_iteration(node, 0, 0.0, 0.0, table_path=progress_table)
            best_path = out_dir / "best_get_tree_model.py"
            best_path.write_text(baseline_src)
            _report_run_summary(node, t_start, 0, 0.0, 0.0, 0.0, 0.0, 0)
            _finalize(out_dir, node, [{"iteration": 0, "solved": True}], 0, 0.0,
                      src_path, entry, apply_best, best_path)
            return best_path

        # ── curriculum ordering ──
        ordered = oracle.cases_by_volume()
        if curriculum:
            active = [ordered[0].tag]
        else:
            active = [c.tag for c in oracle.cases]
        oracle.set_active_cases(active)

        archive = Archive()
        # seed the archive/best with the baseline (re-scored locally on the active set)
        base_eval = oracle.evaluate_local(baseline_src) if oracle_mode != "docker" else oracle._eval_from_records(d0["records"])
        archive.record(baseline_src, base_eval)
        best = {"iteration": 0, "source": baseline_src, "score": base_eval.score, "eval": base_eval}
        best_feedback = oracle.format_feedback(base_eval)
        best_messages = None  # memory lineage
        history = [{"iteration": 0, "score": round(base_eval.score, 2), "n_active": len(active)}]

        # baseline delta vs tree model (worst abs delta + avg normalized %) -> table.
        base_maxabs, base_avgnorm = progress.delta_ratios(base_eval)
        progress.record_iteration(node, 0, base_maxabs, base_avgnorm, table_path=progress_table)
        last_iter = 0

        print(f"\nbaseline local score (active set)={round(base_eval.score, 2)}")
        print(f"baseline delta vs rtlsim: max_abs={base_maxabs:g} avg_norm={base_avgnorm:.2f}%")
        print(best_feedback)

        for i in range(1, max_iterations + 1):
            n_active = len(oracle._active())
            print(f"\n{'=' * 60}\niteration {i}  (active cases: {n_active}/{len(oracle.cases)}, "
                  f"best score {round(best['score'], 2)})\n{'=' * 60}")

            curriculum_note = (
                f"  [curriculum: solve these first; more will be added as you succeed]"
                if curriculum and n_active < len(oracle.cases) else ""
            )
            extra = RETRY_BLOCK.format(
                score=round(best["score"], 2), best=best["source"].strip(), feedback=best_feedback
            )
            if recombine:
                partials = archive.partials(recombine)
                if len(partials) >= 2:
                    blocks = "\n".join(f"--- partial {j+1} ---\n```python\n{p.strip()}\n```"
                                       for j, p in enumerate(partials))
                    extra += RECOMBINE_BLOCK.format(partials=blocks)

            task = TASK_TEMPLATE.format(
                node=node, hls_ref=hls_ref, rtl_ref=rtl_ref, inputs_dir=INPUTS_DIR,
                n_active=n_active, curriculum_note=curriculum_note,
                baseline=baseline_src.strip(), extra=extra,
            )

            models = _temperature_portfolio(base_model, parallel)
            seeds = [best_messages if (memory and w == 0) else None for w in range(parallel)]
            tpath = lambda w: os.path.join(AGENTS_DIR, f"{safe_node}-iter{i:03d}-pop{w}.log")

            if parallel == 1:
                worker_results = [_run_worker(0, task, models[0], workspace, max_turns,
                                              oracle, archive, seeds[0], bash_budget=bash_budget,
                                              transcript_path=tpath(0))]
            else:
                with ThreadPoolExecutor(max_workers=parallel) as ex:
                    futs = [
                        ex.submit(_run_worker, w, task, models[w], workspace, max_turns,
                                  oracle, archive, seeds[w], bash_budget=bash_budget,
                                  transcript_path=tpath(w))
                        for w in range(parallel)
                    ]
                    worker_results = [f.result() for f in futs]

            # Pick the global best over the archive, scored on the CURRENT active
            # set so comparisons are apples-to-apples even as --curriculum widens
            # it. Re-score each candidate locally on the active set; only fall
            # back to its stored (docker-escalated) result for sibling-method
            # candidates that can't be scored locally -- so helper-method
            # candidates still carry a faithful score without breaking curriculum,
            # where stored scores (taken on a smaller active set) would otherwise
            # let a stale 1-case tree win forever.
            iter_best = None
            for source, r in archive.all:
                if oracle_mode == "docker":
                    ev = r
                else:
                    ev = oracle.evaluate_local(source)
                    if oracle._needs_docker_escalation(ev):
                        ev = r  # sibling-method candidate: keep its faithful stored score
                if iter_best is None or ev.score < iter_best[0]:
                    iter_best = (ev.score, source, ev)
            iter_score, iter_source, iter_eval = iter_best

            (cand_dir / f"iter_{i:03d}.py").write_text(iter_source)
            if iter_score < best["score"]:
                best = {"iteration": i, "source": iter_source, "score": iter_score, "eval": iter_eval}
            best_feedback = oracle.format_feedback(iter_eval)
            # carry forward the message history of worker 0 as the memory lineage
            if memory:
                best_messages = worker_results[0][0]
            last_iter = i
            iter_maxabs, iter_avgnorm = progress.delta_ratios(iter_eval)
            progress.record_iteration(node, i, iter_maxabs, iter_avgnorm, table_path=progress_table)
            history.append({"iteration": i, "score": round(iter_score, 2), "n_active": n_active,
                            "max_abs_delta": round(iter_maxabs, 4),
                            "avg_normalized_delta_pct": round(iter_avgnorm, 4)})
            print(f"\niter {i}: best-in-archive score={round(iter_score, 2)} solved_active={iter_eval.solved} "
                  f"delta vs rtlsim: max_abs={iter_maxabs:g} avg_norm={iter_avgnorm:.2f}%")
            print(best_feedback)

            # ── optional analyzer pass: structural advice for next round ──
            if analyzer and not iter_eval.solved:
                a_task = ANALYZER_TEMPLATE.format(
                    node=node, inputs_dir=INPUTS_DIR, hls_ref=hls_ref, rtl_ref=rtl_ref,
                    candidate=iter_source.strip(), feedback=best_feedback,
                )
                try:
                    a_msgs = [{"role": "system", "content": TAV_SYSTEM_PROMPT},
                              {"role": "user", "content": a_task}]
                    advice, _ = run_agent_with_history(
                        a_msgs, base_model, workspace / "analyzer", max_turns=max_turns,
                        transcript_path=os.path.join(AGENTS_DIR, f"{safe_node}-iter{i:03d}-analyzer.log"),
                    )
                    best_feedback += f"\n\nAnalyzer advice:\n{advice}"
                except Exception:
                    print(f"[analyzer] crashed:\n{traceback.format_exc()}")

            # ── solved on the active set? confirm + maybe widen curriculum ──
            if iter_eval.solved:
                if curriculum and n_active < len(oracle.cases):
                    active = [c.tag for c in ordered[:n_active + 1]]
                    oracle.set_active_cases(active)
                    print(f"[curriculum] active set widened to {len(active)} case(s).")
                    continue

                # full set solved locally -> confirm in docker (source of truth)
                if docker_confirm and oracle_mode != "docker":
                    print("[confirm] candidate solves all cases locally; confirming in docker...")
                    (workspace / CANDIDATE_FILENAME).write_text(best["source"])
                    dconf = oracle.evaluate_docker(str(workspace / CANDIDATE_FILENAME))
                    if dconf["solved"]:
                        print(f"\nDOCKER CONFIRMED solved on iteration {i}.")
                        history.append({"iteration": i, "docker_confirmed": True})
                        break
                    # local said solved, docker disagrees -> oracle drift; feed it back
                    dfb = oracle.format_feedback(oracle._eval_from_records(dconf["records"]))
                    print(f"[confirm] docker DISAGREES (local oracle drifted). docker score="
                          f"{round(dconf['score'], 2)}. Feeding the real delta back.")
                    best_feedback = (
                        "NOTE: your tree matched the local oracle but the REAL docker validator "
                        "still shows deltas below. Trust this docker feedback:\n" + dfb
                    )
                else:
                    print(f"\nSolved on iteration {i} (oracle={oracle_mode}).")
                    break
        else:
            print(f"\nStopped after {max_iterations} iterations.")

        best_path = out_dir / "best_get_tree_model.py"
        best_path.write_text(best["source"])
        final_maxabs, final_avgnorm = progress.delta_ratios(best["eval"])
        _report_run_summary(node, t_start, last_iter, base_maxabs, base_avgnorm,
                            final_maxabs, final_avgnorm, best["iteration"])
        _finalize(out_dir, node, history, best["iteration"], best["score"],
                  src_path, entry, apply_best, best_path)
        return best_path
    except Exception:
        print(traceback.format_exc())
        raise SystemExit(1)
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        log_fh.close()


def _report_run_summary(node, t_start, iterations, base_maxabs, base_avgnorm,
                        final_maxabs, final_avgnorm, best_iter):
    """Print the end-of-run summary for one node: wall-clock minutes, iteration
    count, and the final delta (worst absolute delta + average normalized delta
    %, tree model vs rtlsim) against the baseline."""
    minutes = (time.monotonic() - t_start) / 60.0
    print(f"\n{'=' * 60}\nRUN SUMMARY: {node}\n{'=' * 60}")
    print(f"  duration:   {minutes:.2f} min")
    print(f"  iterations: {iterations}  (best from iteration {best_iter})")
    print(f"  delta vs rtlsim  (max_abs = worst |rtlsim-model|; "
          f"avg_norm = mean |rtlsim-model|/rtlsim):")
    print(f"    baseline: max_abs={base_maxabs:g}  avg_norm={base_avgnorm:.2f}%")
    print(f"    final:    max_abs={final_maxabs:g}  avg_norm={final_avgnorm:.2f}%")
    print(f"  shared progress table: {progress.DEFAULT_TABLE}")


def _finalize(out_dir, node, history, best_iter, best_score, src_path, entry, apply_best, best_path):
    (out_dir / "history.json").write_text(json.dumps(
        {"node": node, "best": best_iter, "best_score": best_score, "history": history}, indent=1))
    if not os.path.isabs(src_path):
        src_path = os.path.join(tav_eval.FINN_ROOT, src_path)
    tav_eval.restore_original(src_path)
    test_file = entry["test"].partition("::")[0]
    if not os.path.isabs(test_file):
        test_file = os.path.join(tav_eval.FINN_ROOT, test_file)
    tav_eval.restore_original(test_file)
    if apply_best:
        tav_eval.replace_function(src_path, str(best_path))
        print(f"applied best candidate (iteration {best_iter}) to {src_path}")
    print(f"\nbest candidate: {best_path}")
    print(f"history:        {out_dir / 'history.json'}")


def _split_nodes(node_args) -> list[str]:
    """Parse the positional node argument(s) into a list of node names. Multiple
    nodes are separated by commas (whitespace around them is fine), so all of
    ``MVAU,FMPadding`` / ``MVAU, FMPadding`` / ``"MVAU, FMPadding"`` work."""
    raw = " ".join(node_args) if isinstance(node_args, (list, tuple)) else str(node_args)
    return [n for n in re.split(r"[\s,]+", raw.strip()) if n]


def _run_node_process(node, kwargs):
    """Child-process entry point for one node's optimization loop (multi-node
    mode). Re-loads .env so the worker has API keys, then runs the loop."""
    load_dotenv()
    run_loop(node, **kwargs)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", nargs="+",
                    help="one or more FINN node names (tav_eval.py --list). Separate multiple "
                         "with commas to optimize them in parallel, e.g. 'MVAU, FMPadding'.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--max-iterations", type=int, default=20)
    ap.add_argument("--max-turns", type=int, default=60, help="agent tool-call turns per worker per iteration")
    ap.add_argument("--parallel", type=int, default=4,
                    help="population size: N agents per iteration at spread temperatures")
    ap.add_argument("--bash-budget", type=int, default=25, metavar="N",
                    help="max bash/run (source-reading) calls per worker per iteration before it "
                         "is told to iterate via eval_tree_model instead; 0 = unlimited (default 25)")
    ap.add_argument("--oracle", choices=["local", "docker", "both"], default="local",
                    help="self-eval backend (local = fast, docker-confirmed; default: local)")
    ap.add_argument("--curriculum", action="store_true",
                    help="start on the smallest test case, widen one case at a time")
    ap.add_argument("--memory", action="store_true",
                    help="carry the best lineage's conversation across iterations")
    ap.add_argument("--analyzer", action="store_true",
                    help="run a second agent each round for structural advice")
    ap.add_argument("--recombine", type=int, default=0, metavar="K",
                    help="feed back up to K archived partial solutions to recombine")
    ap.add_argument("--no-docker-confirm", dest="docker_confirm", action="store_false",
                    help="declare solved on the local oracle without a docker confirmation (faster, unverified)")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
                    help="override the model's reasoning effort")
    ap.add_argument("--feedback-cap", type=int, default=120, metavar="PAIRS",
                    help="cap (in RLE run pairs) for every vector shown in detail "
                         "(TARGET/YOURS/DELTA); the RLE front-loads the divergence onset (default 120)")
    ap.add_argument("--feedback-budget", type=int, default=40000, metavar="CHARS",
                    help="char budget for the detailed section of one feedback message; "
                         "bounds prompt size for nodes with many/long cases (default 40000)")
    ap.add_argument("--feedback-detail-cases", type=int, default=10, metavar="N",
                    help="max number of failing cases shown in full detail per feedback "
                         "(smallest-vector-first); the rest are summarized (default 10)")
    ap.add_argument("--baseline", help="seed get_tree_model.py (default: tav_eval's example for this node)")
    ap.add_argument("--src", help="override the node source file path")
    ap.add_argument("--test", help="override the pytest nodeid to run")
    ap.add_argument("--extra-tests", action="store_true", help="also run registry extra_tests")
    ap.add_argument("--cache-dir", help="rtlsim reference cache dir (default <repo>/cached_models)")
    ap.add_argument("--out", help="output directory (default under $FINN_HOST_BUILD_DIR/tav_bespoke)")
    ap.add_argument("--apply-best", action="store_true", help="splice the best candidate into the node source at the end")
    ap.add_argument("--log", default=None, help="log file (default under outputs/)")
    ap.add_argument("--progress-table", default=None,
                    help=f"shared progress table path (default {progress.DEFAULT_TABLE}); all "
                         "concurrently-optimized nodes append to the same file")
    args = ap.parse_args()

    nodes = _split_nodes(args.node)
    if not nodes:
        raise SystemExit("at least one node is required (see tav_eval.py --list)")

    base_ws = Path(args.workspace).resolve()
    # kwargs shared by every node loop (everything except per-node node/workspace/log).
    common = dict(
        model=args.model,
        max_iterations=args.max_iterations,
        max_turns=args.max_turns,
        parallel=args.parallel,
        bash_budget=args.bash_budget,
        oracle_mode=args.oracle,
        curriculum=args.curriculum,
        memory=args.memory,
        analyzer=args.analyzer,
        recombine=args.recombine,
        docker_confirm=args.docker_confirm,
        reasoning_effort=args.reasoning_effort,
        feedback_cap=args.feedback_cap,
        feedback_budget=args.feedback_budget,
        feedback_detail_cases=args.feedback_detail_cases,
        baseline=args.baseline,
        src_override=args.src,
        test_override=args.test,
        include_extra_tests=args.extra_tests,
        cache_dir=args.cache_dir,
        apply_best=args.apply_best,
        out_dir=args.out,
        progress_table=args.progress_table,
    )

    if len(nodes) == 1:
        run_loop(nodes[0], workspace=base_ws, log_path=args.log, **common)
        return

    # ── multiple nodes: one OS process each, running in parallel ──
    # Separate processes (not threads) because run_loop redirects the global
    # sys.stdout/stderr for its log; each node needs its own isolated workspace,
    # log file and stdout. The docker container runs all the per-node pytest
    # evals concurrently (each node splices a different FINN source file). The
    # shared progress table is flock-guarded, so all processes update it safely.
    import multiprocessing as mp

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ctx = mp.get_context("spawn")
    procs = []
    print(f"optimizing {len(nodes)} nodes in parallel: {', '.join(nodes)}")
    for n in nodes:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", n)
        kw = dict(common)
        kw["workspace"] = base_ws / safe
        kw["log_path"] = os.path.join(OUTPUTS_DIR, f"tree-model-run-{safe}-{ts}.log")
        p = ctx.Process(target=_run_node_process, args=(n, kw), name=f"opt-{n}")
        p.start()
        procs.append((n, p))
        print(f"  [{n}] pid={p.pid} log={kw['log_path']}")

    failed = []
    for n, p in procs:
        p.join()
        if p.exitcode != 0:
            failed.append((n, p.exitcode))
    print(f"\nall {len(nodes)} node loops finished. shared progress table: "
          f"{args.progress_table or progress.DEFAULT_TABLE}")
    if failed:
        print("FAILED nodes: " + ", ".join(f"{n} (exit {c})" for n, c in failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
