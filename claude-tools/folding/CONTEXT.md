# Folding — resource-aware folding optimizer

## Goal
Replace/augment FINN's greedy `SetFolding` with an optimizer that picks per-layer
PE/SIMD (and related folding params) to hit a throughput target **while respecting
resource budgets** (LUT/BRAM/DSP/URAM), and that plays well with the DWC and FIFO
sizing passes.

## Branches
- Feature branch: **`feature/set-folding-optimizer`**; active dev on **`dev-set-folding-optimizer`**.
- Depends on the generalized **[DWC](../dwc/CONTEXT.md)** (folding wants to pad channel
  counts to rounder values, which needs a padding-capable DWC) and feeds
  **[FIFO sizing](../fifo-sizing/CONTEXT.md)** (folding sets stream widths → DWC
  insertion → FIFO depths).

## Key source files
- `src/finn/transformation/fpgadataflow/set_folding.py`
  - **Current `SetFolding`** (~line 72): greedy — for each node walk divisors of max
    parallelism, stop at first PE/SIMD meeting `target_cycles_per_frame`. MVAU: raise
    SIMD until weight width ≤ `mvau_wwidth_max` (36), then PE. VVAU/Pool/SWG paired.
    `two_pass_relaxation` rebalances to the bottleneck.
  - **New `Optimizer`**: cost functions + **simulated annealing**, resource-aware,
    param whitelist, optional DWC + FIFO heuristics, sizes FIFOs mid-optimization
    (`insert_and_size_fifos()`) rather than post-hoc.
- Config via `target_fps` or `folding_config_file` JSON.

## Status (2026-07-22)
- Integrated onto latest dev as `dev-folding` / `folding-pr` (= dev + optimizer +
  `parallel_window` fix). Enable in a build: `cfg.folding_style="optimizer"` +
  `cfg.target_fps` (target_fps=None ⇒ SetFolding is SKIPPED; there is no
  `target_cycles_per_frame` build-config field — it's derived from target_fps).
- **`parallel_window` search-space fix** (committed): the SWG↔MVAU pairing capped
  `mvau_simd<=ifm_channels` and only set `parallel_window=1` when `mvau_simd==ifm_channels`,
  so the JSON's high-throughput region (full kernel window, MVAU SIMD up to MW) was
  unreachable. Fix: allow `mvau_simd` a multiple of `ifm_channels` (see
  `_pair_swg_with_mvau`). Optimizer now matches the reference JSON's throughput.
  **Deferred:** SA convergence still doesn't hit the tightest targets (vgg10 1538 vs 1028)
  and is non-monotonic across attempts — retain-best-config / strict target-meeting is next.
- **Eval harness `folding_eval.py`** (this folder): caches the shared pre-folding checkpoint
  per model, re-runs each method (`json`/`optimizer`/`naive`) → fold → **regular OOC synth**
  (no P&R) → real LUT/FF/BRAM/DSP from `finn_design_partition_util.rpt`; per-method synth
  cached in stable dirs. gtsrb result: optimizer matches json throughput (33020 cyc) at
  comparable resources (+6% LUT, -11% FF). Run: `python claude-tools/folding/folding_eval.py
  gtsrb Pynq-Z1 json,optimizer 100`.

## Older status / knobs
- The optimizer runs on ~4/5 finn-examples models; SA effort is controlled by a
  `folding_effort` knob (characterized separately).
- resnet50 was blocked by a partition cycle at `step_create_dataflow_partition` —
  fixed on `resnet50-partition-fix` via `MoveTransposePastEltwiseBinary` (also unblocks
  fifo sizing / branch DWC paths).

## Tooling (add here when working on folding)
- `folding_sweep.py` — auto-folder (SetFolding) stress test: drop the folding config,
  set `target_fps`, run through `step_generate_estimate_reports`, report which op types
  break. Injects `step_target_fps_parallelization` for the models whose custom step
  lists omit it. (Lives in the repo `tools/` when present; copy into this folder while
  iterating so it travels with `claude-tools`.)

## Gotchas
- Never auto-fold when doing **sizing** runs — use each model's reference folding JSON
  (fixed PE/SIMD). Auto-fold is only for exercising the optimizer itself.
