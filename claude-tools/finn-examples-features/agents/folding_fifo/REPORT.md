# Report: FIFO-aware folding

Branch `feature/dev/set-folding-optimizer-fifo-aware` (worktree `/tmp/finn_dev_lstasytis/worktrees/folding-fifo`,
Vivado 2024.2), three commits on top of `8ea4ed94b`:

| commit | what |
|---|---|
| `ec9929b36` | SetFolding optimizer: FIFO-economical foldings (SWG folded on its own, FIFO estimate in the cost model) |
| `e30a3f6ba` | SetFolding optimizer: FIFO-aware mode on the analytic sizer (repair of `enable_folding_fifo_heuristic`) |
| `1f9713080` | Folding tests: FIFO estimate, FIFO-aware mode, unsizeable graphs |

Only `set_folding.py` and `tests/fpgadataflow/test_set_folding.py` are touched. Logs and raw results in
`/tmp/finn_dev_lstasytis/agents/folding_fifo/logs/` (`baseline_cmp.log`/`base_cmp3.log` = old code, `final_cmp.log` = new,
`tightness_*`, `quick_folding_flows.log`, `full_cnv.log`); PLAN.md has the step-by-step history.

## 1. Diagnosis: why the optimizer's foldings needed 2.4x the FIFO storage

Measured with the analytic sizer (quick mode, ZCU104, committed JSON's target). The gap was a cnv-w1a1 thing:
naive 14.8 kbit vs optimizer 36.1; on cnv-w2a2 the optimizer was already below the naive (23.3 vs 25.5) and on
tfc-w2a2 equal (7.8 vs 7.4). The per-edge picture of the old cnv-w1a1 folding:

| kbit | FIFO | edge | why |
|---|---|---|---|
| 13.4 | 107 x 128 bit | `SWG_4 (SIMD 128, parallel_window, 146 cyc/frame) -> DWC 1152->128 (900) -> MVAU_3 (SIMD 128, PE 16, 7200)` | producer chain 8x faster than the consumer |
| 7.5 | 40 x 192 | `SWG_3 (pw, 198) -> DWC 576->192 (432) -> MVAU_2 (SIMD 192, PE 8, 6912)` | 35x faster |
| 4.9 | 13 x 384 | `SWG_6 (pw, 27) -> DWC -> MVAU_4 (SIMD 384, PE 1, 6912)` | 256x faster |
| 2 x 2.2 | 2 x 1152 | `SWG_4/SWG_6 -> DWC` | 1152-bit words at minimum depth |

Two causes in the optimizer, neither visible to its cost model:

1. **The SWG pairing rule tied the SWG's SIMD to the MVAU's** (`swg_simd = mvau_simd`, or the whole window with
   `parallel_window`). An SWG emits `OFM^2 * K^2 * C / SIMD` words per frame, the MVAU takes `MH/PE` times as
   many cycles for them, so every SWG was exactly MH/PE times faster than its consumer and ran that far ahead.
   The sizer charges a chain edge `max(peak - slack, min(peak, demand floor))`; with the peak that large the
   floor `words * (1 - t_consumer / T_frame)` bites (MVAU_3: 900 words x (1 - 7200/8167) = 107). The naive
   folder avoids this by accident: it folds the SWG to just meet the target and lets a cheap DWC widen the stream
   (e.g. SWG SIMD 16 -> DWC -> MVAU SIMD 576).
2. **`parallel_window` was switched on already at MVAU SIMD == IFMChannels** (`>=`), producing 1152-bit
   SWG words, a DWC that only splits them, and two FIFOs of that width for nothing.

What matters for resources is not kbit but width: with fifo.sv's cost model a 2-deep 1152-bit FIFO costs 1160
LUT, the 107 x 128 one 433. Including FIFOs and DWCs the old optimizer folding was still cheaper than the
naive one (90.2 k vs 98.4 k LUT), so the real objective is the total.

## 2. Fixes (all in `set_folding.py`)

**Goal 2, FIFO-economical foldings (`ec9929b36`):**
* The SWG is folded on its own: `_pair_swg_with_mvau`'s meta value is `(mvau_simd, swg_simd, parallel_window)`,
  any SWG SIMD dividing the (padded) channel count up to the MVAU SIMD, `parallel_window` only when the MVAU
  SIMD exceeds the channel count, a DWC between the two where the widths differ (the DWC heuristic's old
  exclusion of SWG -> MVAU pairs is gone; `dwc_cycles` already priced them). For a padded channel count only
  SIMDs whose padding fits in a word are offered (InsertDWC's `rtl_supported` rule). The padded search space
  stays a superset (ascending channel counts, de-duplication by folding factors).
* `Optimizer.fifo_resources`: the FIFOs the build will insert, estimated without sizing: one per edge (two around
  a DWC, the graph's I/O FIFOs at depth 5), depth = the producer chain's run-ahead `words * (1 - t_up / t_c)`
  (`t_up` = slowest node upstream, a node paced by a slower one upstream runs no faster) capped by the sizer's
  demand floor `words * (1 - t_c / T) + burst`, at least 2; costed with fifo.sv's own model
  (`finn.util.resource_models._fifo_cost/_resolve`, feature-probed; width x stages fallback). Within ~10 % of
  the sizer's LUTs on both the old and the new cnv-w1a1 folding (5665 vs 6101, 2275 vs 2731). It is part of
  the annealer's cost, the polish's cost and `FoldResult.cost` (used by `_better`), always on (`fifo_proxy`).
* The cheapest node-wise seed memoizes node estimates (the pair groups grew ~5x) and takes the slowest SWG
  that meets the target by construction.

**Goal 1, FIFO-aware mode repaired (`e30a3f6ba`):** `insert_and_size_fifos` calls the analytic sizer's
transformations directly (`DeriveTokenAccessVectors(strategy="tree_model", num_workers=1)`, `DeriveFIFOSizes`,
`InsertFIFO`, `cleanup_characterization`), feature-probed through `analytic_fifo_sizer()` (dev's module has no
`DeriveTokenAccessVectors`): no `DataflowBuildConfig`, no `split_large_fifos`/TAV enums. Synthesis-free by
construction: it declines (`None` + warning) on a tree without the sizer and for any graph with a HW node that
has no tree model (a DWC with a non-integer width ratio), and `_score` then uses the FIFO estimate for that
candidate so it is not scored cheaper than the sized ones. Sidecar `.npy` files go to one work dir per
SetFolding call (removed afterwards; the first version left 7860 `tav_*` dirs behind). `auto_fifo_strategy`
is still accepted (the builder passes it) and only warned about. It reproduces the build's sizing exactly
(36.1 kbit, 37 FIFOs on the old folding) at 0.65-0.8 s per call on cnv-w1a1, which the polish pays per
candidate: 30-47 s per fold against 10-18 s with the estimate, so the mode stays opt-in; the estimate is
the default.

## 3. Before / after (ZCU104, committed JSON's target, quick mode, `heuristic_analytical`; sized models)

Budget: 161 k LUT, 488 BRAM_18K, 74 URAM. Peak = max(used/budget); sum = the optimizer's objective
(sum of used/budget over LUT, BRAM, URAM). Totals include FIFOs and DWCs.

| model | folding | cycles | FIFO kbit | FIFO LUT | DWC LUT | LUT | BRAM | URAM | peak | sum | fold s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnv-w1a1 | naive | 8167 | 14.8 | 3531 | 2690 | 98426 | 84 | 14 | 0.61 | 0.97 | 0 |
| cnv-w1a1 | old optimizer | 8108 | 36.1 | 6101 | 1369 | 90227 | 86 | 12 | 0.56 | 0.90 | 6.6 |
| cnv-w1a1 | **new optimizer** | 8167 | **15.5** | 2731 | 1705 | 104258 | 57 | 4 | 0.65 | **0.82** | 12.2 |
| cnv-w1a1 | new optimizer+fifo | 8167 | 15.6 | 2763 | 1819 | 104404 | 57 | 4 | 0.65 | 0.82 | 30.0 |
| cnv-w2a2 | naive | 33020 | 25.5 | 4537 | 3444 | 54159 | 179 | 45 | 0.61 | 1.31 | 0 |
| cnv-w2a2 | old optimizer | 33020 | 23.3 | 6711 | 1975 | 77395 | 165 | 12 | 0.48 | 0.98 | 7.3 |
| cnv-w2a2 | **new optimizer** | 33020 | **17.5** | 2705 | 1780 | 62390 | 146 | 10 | **0.39** | **0.82** | 12.0 |
| cnv-w2a2 | new optimizer+fifo | 32400 | 23.7 | 2016 | 944 | 61455 | 145 | 10 | 0.38 | 0.81 | 40.8 |
| tfc-w2a2 | naive | 9 | 7.4 | 2538 | 2744 | 232378 | 0 | 0 | 1.44 | 1.44 | 0 |
| tfc-w2a2 | old optimizer | 8 | 7.8 | 2874 | 1604 | 119752 | 0 | 0 | 0.74 | 0.74 | 2.0 |
| tfc-w2a2 | **new optimizer** | 8 | 8.0 | 3010 | 1785 | 119879 | 0 | 0 | 0.74 | 0.74 | 2.6 |
| tfc-w2a2 | new optimizer+fifo | 8 | 8.1 | 3034 | 1833 | 119916 | 0 | 0 | 0.74 | 0.74 | 3.2 |

* FIFO storage: cnv-w1a1 36.1 -> 15.5 kbit (1.05x the naive's), FIFO LUT 6.1 k -> 2.7 k; cnv-w2a2 23.3 ->
  17.5 kbit (0.69x naive), 6.7 k -> 2.7 k LUT, and 15 k fewer LUT overall at 19 fewer BRAM; tfc-w2a2 unchanged
  (no SWG). The largest cnv-w1a1 FIFO is now 6.4 kbit (34 x 192, `DWC -> MVAU_2`), then 3.0 kbit (192 x 16).
* cnv-w1a1 total LUT went up (90 -> 104 k) while BRAM/URAM went down (86/12 -> 57/4): the search moved MVAU_4
  from (PE 1, SIMD 384: 2.8 k LUT + 22 BRAM + 6 URAM) to (PE 4, SIMD 576: 20 k LUT in LUTRAM) -- cheaper by the
  optimizer's budget-weighted objective (0.82 vs 0.90 overall; 6 URAM are 8 % of a ZCU104's) but worse in
  peak utilization (0.65 vs 0.56). This is the objective's valuation of URAM/BRAM against LUT, not a FIFO
  effect (the same folding results with the FIFO estimate switched off); see open questions.
* The in-loop real sizing (`optimizer+fifo`) gives the same foldings within noise at 2.5-3.5x the folding
  time; on cnv-w2a2 it trades 6 kbit more storage for 1 k fewer LUT, which is what the objective asks for.
* Folding time (effort 50, one call, target mode): cnv 6.6-7.3 -> 12.0-12.2 s (the pair groups' larger
  enumeration and the FIFO term in every cost evaluation), tfc-w2a2 2.0 -> 2.6 s.

## 4. Guarantees re-checked

* `test_padding_tightness`: ZCU104 5 passed (`tightness_new1_ZCU104.jsonl`), U55C 5 passed
  (`tightness_final_U55C.jsonl`). Against the previous task's final run (which was on the old worktree; this
  branch's reference targets differ slightly, e.g. tfc-w1a1 3/4/9/21/60): cnv-w2a2 utilization is lower at
  every target (0.44 -> 0.37, 0.41 -> 0.34, 0.37 -> 0.30, 0.40 -> 0.34, 0.39 -> 0.35) and its U55C maximize
  result faster (2352 -> 1568 cycles at 0.95); cnv-w1a1 equal or lower except the reference target
  (0.52 -> 0.63, the MVAU_4 point above); TFC/MLP unchanged. The tfc-w1a1 U55C maximize row (2 cycles instead
  of the old worktree's 1) is the integration branch's, not this task's: HEAD's optimizer gives 2 there too.
* `tests/fpgadataflow/test_set_folding.py`: 34 passed, 1 xfailed (pre-existing).
* `FEATURE_FLOW_QUICK=1 test_folding_flow.py` (builder, folded cppsim vs golden): 12 passed (naive and
  optimizer, 3 models x 2 boards).
* Full mode cnv-w1a1: see section 5.

## 5. Full mode, cnv-w1a1 (HLS + stitched-IP rtlsim, ZCU104, optimizer folding, `full_cnv.log`)

| FIFO sizing | verified (cppsim, rtlsim) | est. cycles | rtlsim fps | FIFO kbit | LUT / BRAM / URAM | fold s | sizing s |
|---|---|---|---|---|---|---|---|
| `heuristic_analytical` | yes, yes | 8167 | 12255 | 15.5 | 104258 / 57 / 4 | 12.0 | 2.4 |
| `largefifo_rtlsim` | yes, yes | 8167 | 12255 | 27.6 | 104279 / 57 / 4 | 11.9 | 668 |

Throughput unchanged (12255 fps, the committed folding's rate and the naive's 12255); the rtlsim-sized
storage is 27.6 kbit against 165.8 kbit for the old optimizer folding (naive 26.9 kbit per the task's findings).

## 6. Open questions / notes

* **Objective, not FIFOs, decides cnv-w1a1's LUT count.** The budget-weighted sum values 6 URAM (8 % of the
  ZCU104's 74) like 13 k LUT; the search therefore prefers MVAU_4 in LUTRAM (20 k LUT) over 22 BRAM + 6 URAM,
  which lowers the sum (0.82 vs 0.90) and raises peak utilization (0.65 vs 0.56). If peak utilization (or a
  lexicographic "fit first, then LUT") is the intended objective, that is a change to `cost_model` /
  `_cost`, independent of this task; `prefer_memory="bram"`/`"uram"` already shift it.
* The FIFO estimate is first order: it reproduces the sizer's LUTs within ~10 % on the measured foldings but
  not per edge (the two-frame split, slack reset past the pacer and the throttled cap are not modelled). It is
  there to steer the search; the verdict with real sizing is the opt-in mode.
* `enable_folding_fifo_heuristic=True` is 2.5-3.5x slower (the polish sizes every candidate at 0.65-0.8 s on
  cnv-w1a1) and gives the same foldings within noise here, so it stays opt-in. Foldings with a DWC of
  non-integer width ratio (no tree model) fall back to the estimate in that mode; a build with
  `heuristic_analytical` JIT-synthesizes such a DWC with HLS (not this task's code).
* Folding time rose ~1.7x in target mode (cnv 7 -> 12 s, tfc-w2a2 2.0 -> 2.6 s): the SWG/MVAU pair groups
  enumerate ~5x more combinations in the cheapest seed (node estimates are memoized) and every cost evaluation
  walks the upstream chain for the FIFO estimate.
* The tightness rows differ from the previous task's run also because this branch's reference folding JSON
  cycles differ (DWC cycles counted in words): tfc-w1a1 targets are 3/4/9/21/60 here.
* Housekeeping: the first in-loop sizer left 7860 empty `tav_*` dirs (sidecar files of nodes without a code
  generation dir); fixed and removed. The old `fold_fifo_cmp.py` dies on a failed build (`KeyError:
  'fifo_total_kbit'`); my copy (`scratchpad/fold_fifo_cmp_fifo.py`) tolerates it and logs JSONL.
