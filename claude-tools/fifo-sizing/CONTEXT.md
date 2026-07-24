# FIFO sizing — analytical (tree-model) sizing

## Goal
Size inter-layer FIFOs analytically (no rtlsim) using per-op **tree models** that
produce Token Access Vectors (TAVs), matching the buffer sizes from the paper and the
"live" (optimal) reference — instead of the slow oversize-then-rtlsim approach.

## Branch
- Feature branch: **`feature/analytical-fifo-sizing`**; dev pattern `-dev` branches
  (e.g. `dev-analytical-fifo-sizing`).
- Consumes folding + DWC output (folding sets widths → DWC insertion → FIFO depths).

## Key source files
- `src/finn/transformation/fpgadataflow/derive_characteristic.py` — analytical core:
  - `DeriveTokenAccessVectors` — per-node TAVs (`io_chrc_in/out`, `io_chrc_period`);
    `strategy = "tree_model" | "rtlsim"`.
  - `DeriveFIFOSizes` — Step 1 peak-delta conservative bound, Step 2 relaxation
    (`tav_utilization_strategy`: conservative / aggressive / no_relaxation).
  - `Local/DelayCharacteristicFunctions`, `HandleBranches` (Duplicate/AddStreams),
    `JustInTimeSynthesize` (synth only nodes lacking a tree model).
- `src/finn/custom_op/fpgadataflow/hwcustomop.py` — `get_tree_model()` (overridden per
  op) and the TAV builders. **Optimizing these tree models is the core work.**
- Entry: `step_set_fifo_depths()` in `build_dataflow_steps.py`.

## Key facts (verify before trusting — these drift)
- **TAV = Token Access Vector**: cumulative **transaction-count** curves over 2 periods,
  units are transactions/words **not bits**. Any pairwise TAV comparison across a node
  that changes the token rate (DWCs, pooling/stride) is invalid unless rescaled by the
  width ratio — this was the big **producer→DWC over-sizing bug** (sized against the
  post-DWC consumer instead of the direct DWC consumer, inflating FIFOs ~36×). Fix:
  size against `model.find_consumer(output_name)` (the DWC), keep DWCs see-through only
  for period/relaxation.
- Relaxation: paper Table II `modeled` used **conservative**; the newer Table 3
  heuristic column uses **aggressive**.

## Status
- Solved: gtsrb, vgg10, kws, cybersec (near paper targets). mobilenet is the main gap —
  its over-size is **algorithmic** (isolated-characterization ignores backpressure);
  a global-stretch knob helps mobilenet but under-sizes kws — refinement spec'd.
- resnet50 exercises the branch-sizing path (AddStreams/DuplicateStreams) once the
  partition-cycle blocker is cleared (see [folding](../folding/CONTEXT.md)).

## Tooling (add here when working on fifo sizing)
Repo `tools/` scripts (copy into this folder while iterating so they travel with
`claude-tools`):
- `fifo_sizing_sweep.py` — analytical sweep over finn-examples; `--method
  analytic_model_based | analytic_rtlsim | largefifo_rtlsim`; checkpoint/resume via
  stable `$FINN_BUILD_DIR/sweep_<model>_<method>` dirs.
- `fifo_sizing_regression.py` — pass/fail regression + vs-paper report; **run after any
  sizer change** (the DWC-regression guard).
- `fifo_dist.py` — top FIFOs by producer→consumer op type; `tav_probe.py` — per-FIFO
  TAV/period/DWC-width diagnosis; `fifo_kb.py` — total FIFO KB; `gt_depths.py` — embedded
  config depths.
- Sized model lands at
  `$FINN_BUILD_DIR/sweep_<model>_<method>*/intermediate_models/step_set_fifo_depths.onnx`.

## Recipe (KB-per-model via tree models)
`auto_fifo_depths=True`, `auto_fifo_strategy=ANALYTIC`,
`tav_generation_strategy=TREE_MODEL`, `tav_utilization_strategy=CONSERVATIVE_RELAXATION`;
truncate the step list at `step_set_fifo_depths` (no synth → runs on Vivado 2023.1).

## NEXT: finn-examples benchmark testing (2026-07-22) — not yet started
Goal: show the analytic tree-model FIFO KB is close to the ground truth, vs
`analytic_rtlsim` and `largefifo_rtlsim`, per model. Replicate the folding pattern (see
[`../README.md`](../README.md) START HERE):
1. `git checkout -b dev-analytical-fifo-sizing dev-expanded-finnexamples`;
   `git cherry-pick`/merge `origin/feature/analytical-fifo-sizing` (only ~179 behind dev,
   +23 commits — a merge may be cleaner than cherry-pick). Confirm the analytic strategy is
   wired into the dev build config (`auto_fifo_strategy`/`tav_generation_strategy`).
2. **Re-integrate the cached gtsrb fifo-variant benchmark test** — it existed on our OLD
   `expanded-finnexamples` (commits `1064e50db`/`ed45faaf9`, function `test_fifo_sizing_gtsrb`)
   but NOT on the real add_finnexamples, so it must be re-applied onto the new gtsrb test.
   Its caching: search `$FINN_BUILD_DIR` for a `build_fifo_<model>_` dir; if
   `step_hw_ipgen` is cached set `cfg.steps=["step_set_fifo_depths"]` and re-run only sizing;
   else run the flow (minus deploy) caching at ipgen. Reports `compute_total_model_fifo_size`
   (KB) via `extract_model_config_to_json`. Generalize to all models; compare the 3 methods
   and to `tools/paper_ground_truths.json` (the `modeled` column = target). FIFO sizing does
   NOT need synth, so this is fast; but the harness caches at ipgen so the rtlsim methods can
   reuse the built IP.
3. mobilenet is the known over-size gap; gtsrb/vgg10/kws/cybersec are solved (near GT).
