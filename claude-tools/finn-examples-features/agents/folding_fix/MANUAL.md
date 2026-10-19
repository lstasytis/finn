# Task manual: make the folding optimizer reliable, and channel padding a strict win

You are an agent working in a FINN (FPGA dataflow compiler) repository. FINN maps a
quantized neural network to a pipeline of hardware layers; **folding** chooses each
layer's parallelism (PE/SIMD), which sets the pipeline's throughput (cycles per frame,
lower = faster) and its resource cost (LUT/BRAM/URAM/DSP). Your only task is the one
below; stay inside it.

## Goal

`SetFolding(style="optimizer")` in `src/finn/transformation/fpgadataflow/set_folding.py`
is a resource-aware simulated-annealing folder:

* given `target_cycles_per_frame`, it should meet that throughput at minimum cost;
* given `None`, it should maximize throughput within the device resource budget;
* with `folding_maximum_padding > 0` it may pad a layer's channel count (e.g. 10 -> 12) to
  unlock finer PE/SIMD choices; DWCs between padded and unpadded layers pad/crop each row.
  Padding exists to give **tighter** foldings: closer to the target at lower cost, or more
  throughput at full utilization.

A benchmark shows it does not do this reliably. Fix the optimizer so that:

1. **Targets are met whenever feasible.** If the unpadded optimizer, the naive folder, or
   any folding you can construct meets a target within budget, the optimizer must not
   return a folding that misses it.
2. **Padding never makes a result worse.** The padded search space contains the unpadded
   one, so a padded run must be at least as good: target mode, the same target met at no
   more cost (within 2 %); maximize mode, no more cycles (within 2 %), within budget.
3. **Padding shows its benefit** where channel counts have few divisors (the benchmark
   already shows e.g. tfc-w1a1 at a 14-cycle target: padded 37 k LUT vs unpadded 65 k, but
   the padded run then misses the target). Explain in your report where padding helps and
   by how much, with numbers from the benchmark.
4. Search time stays reasonable: report folding time per model before and after (the
   default `folding_effort` in builds is 50). Doubling it is acceptable if it buys the
   guarantees above; say so.

## Evidence (baseline, ZCU104, seed 0, folding_effort 50)

`/tmp/finn_dev_lstasytis/agents/folding_fix/baseline_padding_tightness_ZCU104.jsonl`
(one JSON per model: rows of target_cycles -> unpadded/padded cycles, utilization = max
over resources of used/budget, resources, fits_budget, number of padding DWCs). Examples:

| model | target | unpadded cycles / util | padded cycles / util | problem |
|---|---|---|---|---|
| tfc-w1a1 | 6 | 10 / 0.75 | 13 / 0.55 | padded misses a target unpadded meets |
| tfc-w1a1 | 14 | 10 / 0.41 | 17 / 0.23 | same; unpadded also overshoots the target 1.4x |
| tfc-w2a2 | 56 | 105 / 0.16 | 55 / 0.13 | **unpadded** misses a feasible target |
| cnv-w1a1 | 57169 | 57600 / 0.20 | 54096 / 0.22 | unpadded misses |
| cnv-w1a1 | max | 5184 / 0.81 | 7119 / 0.82 | padded maximize is slower |
| cnv-w2a2 | max | 8167 / 0.96 | 8167 / 1.01 | padded exceeds the budget |

Note: tfc-w1a1 targets 2-3 and cybersecurity-mlp targets 1-3 are infeasible on ZCU104
(over budget either way); that is fine, but report what the optimizer returns there.

## Ground rules

* Work only in your worktree `/tmp/finn_dev_lstasytis/worktrees/folding-fix`, branch
  `feature/dev/set-folding-optimizer-fixes` (based on dev + folding optimizer + DWC
  padding). Never touch `/home/lstasytis/finn_dev_lstasytis/finn` (a long benchmark sweep
  runs there) or other worktrees under `/tmp/finn_dev_lstasytis/worktrees/`, never push,
  never kill processes you did not start.
* `source /tmp/finn_dev_lstasytis/agents/folding_fix/env.sh` in every shell before Python
  (`FINN_ROOT`/`PYTHONPATH` -> your worktree, `FINN_BUILD_DIR` ->
  `/tmp/finn_dev_lstasytis/agents/folding_fix/build`, yours alone).
* The machine (32 cores, 125 GB) is shared: at most `-n 4` with pytest.
* Optimizer changes must stay in `src/finn/transformation/fpgadataflow/set_folding.py`
  (plus folding tests). They will later be moved onto the folding branch, which sits
  *below* the DWC branch, so do not mix them with DWC or test-harness changes in one
  commit. Folding without padding must keep working without the DWC changes.
* Functional correctness is not negotiable: every folding you return must still verify
  (see tools). Known padding rules already in the optimizer: broadcast thresholds are not
  expanded, convolution weights are padded per kernel position, XNOR MVAU inputs are never
  padded (a zero bit means -1 there).
* Commits: small, focused, FINN style (minimal comments, match surrounding code),
  lint-clean. End every commit message with:

  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012VzVo5sDYGjodUk86cC2af
  ```

## Start by writing your own plan

Read the material, reproduce the baseline, then write
`/tmp/finn_dev_lstasytis/agents/folding_fix/PLAN.md`: your diagnosis of each failure mode
(why does SA miss a feasible target? why is the padded search worse when its space is a
superset?), candidate fixes, how you will validate. Keep it updated (mark steps done, note
surprises). Then execute it. Prefer robust structural fixes (e.g. seeding the padded search
with the unpadded solution, keeping the best feasible solution seen, a final repair /
polish pass toward the target) over tuning random knobs.

## Material

* `src/finn/transformation/fpgadataflow/set_folding.py`: `SetFolding` (entry, target vs
  maximize, `_search_folding`, `last_search_report`), `Optimizer` (SA via
  `scipy.optimize.dual_annealing`, cost functions, padding via `allowed_divisors` and
  `PADDING_BOUND_ATTR`, SWG pairing rules, `MetaParameter.apply_value` /
  `update_weight_tensor` / `update_threshold_tensor`).
* The original commit message: `git log --format=%B -1 origin/feature/set-folding-optimizer`.
* Background: `git show origin/claude-tools:claude-tools/folding/CONTEXT.md`.
* Unit tests: `tests/fpgadataflow/test_set_folding.py`.

## Tools

* **The benchmark that defines success**: `tests/benchmark/test_dwc_padding_flow.py::
  test_padding_tightness` (pure Python, a few minutes for all models of a board):
  `pytest tests/benchmark/test_dwc_padding_flow.py -k "tightness and ZCU104" -n 4`.
  It appends one row per model to `$FINN_BUILD_DIR/feature_flow_results/padding_tightness.jsonl`.
  It asserts exactly goals 1-2 in their padded-vs-unpadded form; also check goal 1
  against the naive folder (`style="naive"`) yourself. Run U55C as well before you finish.
* **Correctness**: `FEATURE_FLOW_QUICK=1 pytest tests/benchmark/test_folding_flow.py
  tests/benchmark/test_dwc_padding_flow.py -k "padding_flow or folding_flow" -n 4` folds via
  the full builder and checks the folded model in cppsim against a golden output (one to
  two minutes per case, no synthesis). Full mode (without `FEATURE_FLOW_QUICK`) adds HLS
  synthesis + stitched-IP rtlsim: use it for a final check on tfc-w2a2 and cnv-w2a2
  only (tens of minutes each; avoid cybersecurity-mlp in full mode, its HLS takes hours).
* `python tests/benchmark/feature_flow_report.py` tabulates the feature-flow results.
* Lint: `V=/tmp/claude-1205/-home-lstasytis-finn-dev-lstasytis-finn/8871d419-69de-49fc-824c-30ba6c67e801/scratchpad/lintenv`;
  `$V/bin/isort <files>; $V/bin/black --line-length 100 <files>; $V/bin/ruff check <files>`.

## Deliverables

1. Commits on `feature/dev/set-folding-optimizer-fixes` touching only `set_folding.py` and
   folding tests (plus, in separate commits, any benchmark-test adjustment you can justify;
   never weaken the two guarantees).
2. `test_padding_tightness` passing on ZCU104 and U55C; folding/padding quick flows
   verifying; `tests/fpgadataflow/test_set_folding.py` passing.
3. `/tmp/finn_dev_lstasytis/agents/folding_fix/REPORT.md`: diagnosis, fixes, before/after
   tables (cycles, utilization, folding time) per model/target, where padding helps and by
   how much, open questions. Your final reply summarises it in a few paragraphs with the
   commit list.
