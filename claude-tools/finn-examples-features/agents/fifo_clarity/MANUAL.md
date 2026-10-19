# Task manual: make the chained-TAV FIFO sizing heuristic understandable

You are an agent working in a FINN (FPGA dataflow compiler) repository. The user is the
author of FINN's analytical FIFO sizing. Your only task is the one below; stay inside it.

## Goal (the user's words, condensed)

The heuristic FIFO sizer that turns **token access vectors (TAVs)** into FIFO depths
(`derive_chained_tav_depths` and `DeriveFIFOSizes` in
`src/finn/transformation/fpgadataflow/derive_characteristic.py`) is "extremely hard to
reason about and understand". Make it **simpler and clearer**:

1. Refactor the chained-TAV pass so the path from TAVs to depths reads as a short sequence
   of named, documented steps. Remove what does not carry its weight; merge what is
   duplicated; give magic numbers names and a reason.
2. Document every heuristic (every rule that is more than "depth = peak occupancy"):
   what it does, why it is needed, and **a concrete example from a real model**, in the
   form: *"heuristic X does Y. Example: in cnv-w1a1 the FIFO between MVAU_hls_2 and
   ConvolutionInputGenerator_rtl_3 is depth A without X; X considers property Z, so it
   becomes depth B, and rtlsim throughput no longer drops (C fps -> D fps)."*
   Measure these numbers, do not guess them.
3. FIFO sizing must **never lose more than 1-2 % throughput** against the reference
   (`largefifo_rtlsim`: oversized FIFOs simulated, trimmed to the observed occupancy).

## Ground rules

* Work only in your worktree: `/tmp/finn_dev_lstasytis/worktrees/fifo-clarity`, branch
  `feature/dev/analytical-fifo-sizing-clarity`. Never touch `/home/lstasytis/finn_dev_lstasytis/finn`
  (the main worktree runs a long benchmark sweep), never check out other branches there,
  never push, never kill processes you did not start.
* `source /tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh` in every shell before Python.
  It points `FINN_ROOT`/`PYTHONPATH` at your worktree and `FINN_BUILD_DIR` at
  `/tmp/finn_dev_lstasytis/agents/fifo_clarity/build` (yours alone).
* The machine (32 cores, 125 GB) is shared with the sweep: use at most `-n 4` with pytest
  and keep `NUM_DEFAULT_WORKERS=4`.
* Behaviour first: the refactor must give **identical depths** on your regression set,
  unless you deliberately simplify a rule. A deliberate change needs evidence: rtlsim
  throughput >= 98 % of the reference and total FIFO storage not worse, on every model of
  the regression set. Record every such change in the docs and in the commit message.
* Commits: small and focused, FINN style (minimal comments, match surrounding code),
  lint-clean (see below). End every commit message with:

  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012VzVo5sDYGjodUk86cC2af
  ```

  Configure nothing globally; the repository already has a local git identity.

## Start by writing your own plan

Before changing code, read the material below, then write
`/tmp/finn_dev_lstasytis/agents/fifo_clarity/PLAN.md`: the heuristics you found (one line
each), how you will build the regression set, the refactoring steps, the experiments for
the documentation examples, and how you will validate. Keep it updated as you go (mark
steps done, note surprises). Then execute it.

## Material to read

* `src/finn/transformation/fpgadataflow/derive_characteristic.py` (~800 lines; the chained
  pass is `derive_chained_tav_depths` + helpers `stream_curves`, `arrival_of`,
  `periodic_peak`, `peak_occupancy`, `causal_writes`, `burst_above_rate`, the two
  `QUIRK` functions, and `DeriveFIFOSizes`).
* `src/finn/builder/build_dataflow_steps.py`, `step_set_fifo_depths` (how the pass is
  invoked: `heuristic_analytical` = TAVs from tree models, `heuristic_rtlsim` = TAVs from
  per-node rtlsim; both then call `DeriveFIFOSizes`).
* `src/finn/custom_op/fpgadataflow/hwcustomop.py` (`get_tree_model`, TAV builders) and
  `src/finn/util/basic.py` (`Characteristic_Node`), only as far as you need to understand
  what a TAV is.
* The commit that produced this file: `git show 55b0b08ce --stat` and its message. It lists
  what was stripped, the hard-wired defaults (slack relaxation 1.0, burst floor, throttled
  cap 256, slack side "up", cap guard "pacer", cap margin 1.25, both frames) and why the
  two QUIRK rules exist. The parent branch with all alternatives is
  `origin/feature/analytical-fifo-sizing`.
* Background notes: `git show origin/claude-tools:claude-tools/fifo-sizing/CONTEXT.md`
  (TAVs are cumulative transaction counts over two periods; DWCs change the token rate).
* Unit tests: `tests/fpgadataflow/test_fifosizing.py` (note: its AUP-ZU3 cases fail the
  config check on this machine's Vivado 2023.1, that is environmental) and the
  `tree_model_test` cases marked `node_tree_modeling` in `tests/fpgadataflow/`.

## Tools you have

* Feature-flow benchmark: `tests/benchmark/test_fifo_sizing_flow.py` with the shared harness
  `tests/benchmark/feature_flow_helpers.py` (read its module docstring). It caches each
  (model, board) after `phase_convert_to_hardware`, resumes from there, sizes the FIFOs with
  a strategy and verifies against a golden I/O pair.
  * `FEATURE_FLOW_QUICK=1` stops after FIFO sizing (seconds per model, no synthesis, no
    rtlsim): use it for the depth regression. The sized model is
    `<output_dir>/intermediate_models/step_set_fifo_depths.onnx` (StreamingFIFO nodes carry
    `depth`; the producer/consumer around each FIFO name the edge). Results rows land in
    `$FINN_BUILD_DIR/feature_flow_results/*.jsonl`; `python tests/benchmark/feature_flow_report.py`
    prints a table.
  * Full mode adds HLS IP generation, stitched IP and rtlsim throughput (minutes to tens of
    minutes per model; synthesized HLS IP is reused across runs via
    `$FINN_BUILD_DIR/featureflow_ipcache.json`, pre-seeded for tfc-w1a1 / cnv-w1a1 on
    ZCU104 and U55C). Only FIFO depths differ between your experiments, so after the first
    run per model only stitching + rtlsim are paid.
  * Models: `FEATURE_MODELS` in the harness (tfc-w1a1, cnv-w1a1, cybersecurity-mlp). Avoid
    cybersecurity-mlp in **full** mode: its fully unrolled MVAU takes ~4.5 h of HLS. You may
    add more finn-examples models for the regression set (e.g. gtsrb, kws, vgg10-radioml,
    mobilenet_v1, and bnn-pynq cnv-w2a2/tfc-w2a2) in quick mode; the per-model
    `tests/benchmark/<model>/test_build_*.py` hold their configs. Keep such harness changes
    in a separate commit. Use board ZCU104.
  * Reference throughput: run the same model with `auto_fifo_strategy=largefifo_rtlsim`
    (slow: it synthesizes and simulates internally) or compare against rtlsim with all
    FIFOs made very deep. Note: on this branch the estimated cycles of DWC nodes are
    inflated (a known dev bug, fixed on another branch); trust rtlsim, not estimates.
* For examples ("depth A without heuristic X"), ablate a heuristic in a scratch copy of the
  code (or behind a temporary private flag you remove afterwards), size + rtlsim, compare.
* Lint: `V=/tmp/claude-1205/-home-lstasytis-finn-dev-lstasytis-finn/8871d419-69de-49fc-824c-30ba6c67e801/scratchpad/lintenv`;
  `$V/bin/isort <files>; $V/bin/black --line-length 100 <files>; $V/bin/ruff check <files>`
  (pre-commit itself does not run here).

## Deliverables

1. Commits on `feature/dev/analytical-fifo-sizing-clarity`: the refactored sizer (behaviour
   identical on the regression set, or justified changes), any regression test you add
   (e.g. a quick depth-regression test over fixture models) and the documentation.
2. Documentation page `docs/finn/fifo_sizing_heuristics.rst` (FINN docs are reST; link it
   from the docs index next to the existing FIFO sizing material if there is a natural
   place) plus clear module/function docstrings: what a TAV is, the chained pass step by
   step, then one section per heuristic with its measured example(s).
3. `/tmp/finn_dev_lstasytis/agents/fifo_clarity/REPORT.md`: what you simplified (before/after
   line counts and structure), the heuristic list with one-line explanations, the
   regression set and its results (depths identical / changed with throughput evidence),
   open questions for the user. Your final reply should summarise the same in a few
   paragraphs and give the commit list.
