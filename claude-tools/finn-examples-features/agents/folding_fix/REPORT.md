# Report: folding optimizer reliability and channel padding

Branch `feature/dev/set-folding-optimizer-fixes` (worktree `/tmp/finn_dev_lstasytis/worktrees/folding-fix`),
commits on top of `9ae0d9f9a`:

| commit | what |
|---|---|
| `0173d242d` | SetFolding optimizer: meet the target on the graph the build carries (DWC-aware cost, hard target, as-inserted scoring, repair/polish, seeded padded search, sorted pairing rules) |
| `9da4e5304` | SetFolding optimizer: start the annealer from the cheapest node-wise folding |
| `be14915e6` | Folding tests: DWC-aware target, padding never worse |
| `529dc6c0f` | SetFolding optimizer: fold LabelSelect PE like the naive folder |

Only `src/finn/transformation/fpgadataflow/set_folding.py` and `tests/fpgadataflow/test_set_folding.py`
are touched; no benchmark-test change was needed. Raw data: `/tmp/finn_dev_lstasytis/agents/folding_fix/logs/`
(`tightness_{baseline,fix1,fix2,fix3}_ZCU104.jsonl`, `tightness_fix{2,3}_U55C.jsonl`, timed sweeps
`before_ZCU104.jsonl` (optimizer + naive) and `after3_ZCU104.jsonl`, pytest logs). PLAN.md has the
step-by-step history.

## 1. Diagnosis

A. **The optimizer never saw the DWCs.** The benchmark (and the build) insert DWCs after folding.
   `StreamingDataWidthConverter.get_exp_cycles` is `in_words + out_words` for a non-integer width
   ratio and for every padding/cropping DWC, i.e. up to twice the cycles of either neighbour; the
   cost model used `max(node cycles)` of the partition only and `_fold_at` scored cycles and
   resources on the DWC-free graph. tfc-w2a2 at target 56: the optimizer's graph sits at exactly
   56 cycles, the 28->32 bit DWC between Thresholding (PE=14, 56 words) and MVAU_0 (SIMD=16, 49
   words) costs 105. Padded runs suffered most since every pad/crop DWC is of the slow kind
   (tfc-w1a1 6 -> 13 and 14 -> 17 cycles through a 28->7 bit cropping DWC in front of
   LabelSelect; cnv-w1a1 maximize 5184 -> 7119) and their DWC LUTs were missing from the budget
   verdict (cnv-w2a2 maximize padded: 1.007 of the budget).
B. **The hard constraint was soft by 2 %** (`value_to_minimize_relaxation = 0.98`): cnv-w1a1
   57169 -> 57600.
C. **Under-converging annealer.** Each 3-node partition has ~77 k discrete points; dual_annealing
   samples ~250 of them (effort 50) starting from the most expensive corner (all-max folding).
   tfc-w1a1 at 14 cycles returned 65.5 k LUT at 8-10 cycles although 14-cycle foldings with half
   the MACs exist (SIMD 56 / PE 64 or 112 / 32 on the 784x64 layer). Local search was not the
   problem: it wastes 10-15 % of the evaluations and `no_local_search=True` gives the same result.
D. **Nothing tied the padded run to the unpadded one**: an independent annealing run in a larger
   space with the same evaluation budget per parameter; neither compared nor seeded.
E. **Superset not guaranteed by construction**: the SWG pairing rules iterated `set(ifm_values)`
   and de-duplicated by MVAU SIMD, so a padded (SIMD, MW) could shadow the unpadded one depending
   on set order (CPython's small-int order hides it today).
F. **LabelSelect PE fixed at 1** (`DO_NOT_FOLD`), so a 10-label model could never go below 10
   cycles/frame; the naive folder folds it (tfc-w1a1 target 6: naive 5 cycles within budget,
   optimizer 10). Not a padding issue but a goal-1 miss against the naive yardstick.

## 2. Fixes (all in `set_folding.py`)

1. `Optimizer.dwc_cycles(model, node)`: cycles of the DWC InsertDWC would place in front of a
   node, estimated exactly like `StreamingDataWidthConverter.get_exp_cycles`; the partition's hard
   constraint is `max(node cycles, DWC cycles)`.
2. The throughput target is hard (relaxation 1.0).
3. `SetFolding._score`: each probe's cycles, resources, budget verdict and as-built resources are
   measured on the graph with DWCs inserted and specialized -- what the build carries.
4. `SetFolding._refine` after the annealing passes, on the whole graph with DWCs: *repair* raises
   the folding of what bounds the cycles (the slowest node, or both neighbours of the slowest DWC)
   one step at a time -- cheapest step meeting the target first -- until the target is met
   (terminates: factors only grow, and at full folding no DWC is slower than its neighbours);
   *polish* lowers folding factors one step at a time where that keeps target and budget and does
   not increase the cost (<= 6 whole-graph evaluations per folding parameter; 4-35 ms each).
5. `Optimizer.optimize(initial_guess="cheapest")`: per node, enumerate its own PE/SIMD
   combinations (a few hundred) and start the annealer from the cheapest one meeting the target;
   dual_annealing keeps its start unless it finds better, so a probe is never worse than this
   node-by-node optimum. `initial_guess` also accepts explicit meta values (seeds), and a pass no
   longer resets the parameters of other passes.
6. Padding never worse: with `folding_maximum_padding > 0` the unpadded search runs first, the
   padded one starts from its solution, and `_better` keeps the padded result only if it is no
   worse on every count (target and budget met; peak utilization and weighted cost; in maximize
   mode fewer cycles). Maximize mode replays the unpadded probe sequence padded (each probe seeded
   with the unpadded one at that target) and finally tries to improve on the best unpadded folding
   directly. The unpadded sub-run is seeded and therefore identical to a standalone unpadded run.
7. `sorted(set(ifm_values))` in both SWG pairing rules: the unpadded channel count claims its
   folding factors first, so the padded space is a superset by construction.
8. LabelSelect PE is folded over `Labels` (`PE_MAX_OVERRIDE`; its output width is only the top-K),
   and the polish takes less-folding steps also at equal estimated cost, so this zero-cost
   parameter is folded only as far as the target needs (PE=1 at tfc-w1a1 target 40, PE=2 at 6).
   **Review item**: the original authors excluded it for fmax reasons; the naive folder and the
   build flow have always folded it. One-line revert (`DO_NOT_FOLD`) if that is preferred.

## 3. Results: `test_padding_tightness`, seed 0, effort 50

Baseline: 5 failed (ZCU104). Final: ZCU104 5 passed (499 s), U55C 5 passed (491 s). Each cell is
`cycles / peak utilization` (max over resources of used/budget, DWCs included); `!` = over budget.

### ZCU104, baseline -> final

| model | target | unpadded before | unpadded after | padded before | padded after |
|---|---|---|---|---|---|
| cybersecurity-mlp | 1 | 1 / 3.63! | 1 / 3.63! | 1 / 3.63! | 1 / 3.63! |
| | 3 | 2 / 1.82! | 3 / 1.35! | 5 / 1.24! | 3 / 1.35! |
| | 7 | 5 / 0.78 | 6 / 0.68 | 6 / 0.90 | **6 / 0.63** |
| | 20 | 16 / 0.24 | 20 / 0.20 | 20 / 0.20 | 20 / 0.20 |
| | max | 5 / 0.78 | 5 / 0.78 | 5 / 0.78 | 5 / 0.78 |
| tfc-w1a1 | 2 | 10 / 1.49! | 2 / 1.49! | 10 / 1.49! | 2 / 1.49! |
| | 3 | 10 / 1.49! | 2 / 1.49! | 10 / 1.17! | 2 / 1.49! |
| | 6 | 10 / 0.75 (miss) | 5 / 0.75 | 13 / 0.55 (miss) | **6 / 0.63** |
| | 14 | 10 / 0.41 | 14 / 0.24 | 17 / 0.23 (miss) | 14 / 0.24 |
| | 40 | 32 / 0.12 | 40 / 0.11 | 32 / 0.12 | 40 / 0.11 |
| | max | 10 / 0.38 | 5 / 0.75 | 10 / 0.35 | 5 / 0.75 |
| tfc-w2a2 | 8 | 10 / 0.75 (miss) | 8 / 0.76 | 10 / 1.00 (miss) | 8 / 0.76 |
| | 12 | 10 / 0.75 | 10 / 0.75 | 12 / 0.67 | **12 / 0.54** |
| | 24 | 20 / 0.46 | 20 / 0.39 | 24 / 0.34 | **24 / 0.28** |
| | 56 | 105 / 0.16 (miss) | 56 / 0.13 | 55 / 0.13 | **56 / 0.12** |
| | 160 | 128 / 0.08 | 128 / 0.07 | 128 / 0.09 | 128 / 0.07 |
| | max | 10 / 0.75 | 7 / 0.94 | 10 / 0.68 | **7 / 0.88** |
| cnv-w1a1 | 8167 | 8108 / 0.57 | 8108 / 0.52 | 8192 / 0.55 (miss) | 8100 / 0.52 |
| | 12250 | 10368 / 0.46 | 10800 / 0.40 | 10368 / 0.48 | 10800 / 0.40 |
| | 24501 | 20736 / 0.35 | 24501 / 0.27 | 21600 / 0.37 | 24501 / 0.27 |
| | 57169 | 57600 / 0.20 (miss) | 50176 / 0.21 | 54096 / 0.22 | 50176 / 0.21 |
| | 163340 | 165888 / 0.22 (miss) | 115200 / 0.20 | 165888 / 0.22 (miss) | 115200 / 0.20 |
| | max | 5184 / 0.81 | 4704 / 0.96 | 7119 / 0.82 | 4704 / 0.96 |
| cnv-w2a2 | 33020 | 28224 / 0.54 | 33020 / 0.44 | 32400 / 0.57 | 33020 / 0.44 |
| | 49530 | 41472 / 0.52 | 48600 / 0.41 | 37632 / 0.52 | 48600 / 0.41 |
| | 99060 | 86400 / 0.38 | 97200 / 0.37 | 97200 / 0.38 | 97200 / 0.37 |
| | 231140 | 230400 / 0.40 | 230400 / 0.40 | 207360 / 0.41 | 230400 / 0.40 |
| | 660400 | 460800 / 0.39 | 460800 / 0.39 | 460800 / 0.39 | **576000 / 0.38** |
| | max | 8167 / 0.96 | 7200 / 0.95 | 8167 / 1.01! | **7200 / 0.94** |

Infeasible targets (cybersecurity-mlp 1 and 3, tfc-w1a1 2 and 3 on ZCU104): the optimizer now
returns the fastest folding it has (cybersecurity 3: 3 cycles at 1.35 instead of 2 at 1.82;
tfc-w1a1: 2 cycles at 1.49, the all-max folding), reports `fits_budget=False` and warns. The
padded result equals the unpadded one there (no padded candidate was better on every count).

### U55C, final

| model | target | unpadded | padded |
|---|---|---|---|
| cybersecurity-mlp | 1 / 3 / 7 / 20 / max | 1 / 0.74, 3 / 0.27, 6 / 0.14, 20 / 0.04, 1 / 0.74 | same; 7: 6 / 0.13 |
| tfc-w1a1 | 2 / 3 / 6 / 14 / 40 / max | 2 / 0.30, 2 / 0.30, 5 / 0.15, 14 / 0.05, 40 / 0.02, 1 / 0.60 | same; 6: 6 / 0.13 |
| tfc-w2a2 | 8 / 12 / 24 / 56 / 160 / max | 8 / 0.15, 10 / 0.15, 20 / 0.08, 56 / 0.03, 128 / 0.01, 2 / 0.60 | 12: 12 / 0.11, 24: 24 / 0.06, 56: 56 / 0.02, else same |
| cnv-w1a1 | 8167 ... max | 8108 / 0.11, 10800 / 0.08, 24501 / 0.05, 50176 / 0.03, 115200 / 0.04, 1026 / 0.80 | 8167: 8100 / 0.11, else same |
| cnv-w2a2 | 33020 ... max | 33020 / 0.09, 48600 / 0.06, 97200 / 0.07, 230400 / 0.07, 460800 / 0.07, 2352 / 0.66 | 660400: 576000 / 0.07, else same |

Every target that any folding meets within budget is met (goal 1); the padded result is never
worse (goal 2; it is identical where the padded search found nothing strictly better on every
count, and `_better` then keeps the unpadded folding, which has fewer DWCs).

### Against the naive folder (ZCU104, `before_ZCU104.jsonl`)

The naive folder meets a target the optimizer misses in no row any more (it did on tfc-w1a1
2/3/6 before fix 8; 2 and 3 are over budget for both). Rows where the naive folding has fewer
LUTs are not cheaper: e.g. cnv-w2a2 at 99060 the naive uses 24 k LUT but 211 BRAM and 71 of 77
URAM (peak utilization 0.92, weighted sum 1.49) against the optimizer's 59 k LUT / 156 BRAM / 11
URAM (peak 0.37, weighted sum 0.82); the optimizer trades LUTs for memories by design.

## 4. Where padding helps and by how much

Padding pays where the channel counts have few divisors and the target sits between them --
the TFC layers (784 / 64 / 10 channels) and the MLP (600 / 64 / 1 ... ):

| model | target | unpadded | padded | gain |
|---|---|---|---|---|
| tfc-w2a2 (ZCU104) | 12 | 10 cyc, 121.5 k LUT (0.75) | 12 cyc, 87.4 k LUT (0.54) | -28 % LUT |
| tfc-w2a2 | 24 | 20 cyc, 62.3 k (0.39) | 24 cyc, 44.9 k (0.28) | -28 % |
| tfc-w2a2 | 56 | 56 cyc, 20.9 k (0.13) | 56 cyc, 19.3 k (0.12) | -8 % |
| tfc-w2a2 | max | 7 cyc, 151.6 k (0.94) | 7 cyc, 141.8 k (0.88) | -6 % at equal speed |
| tfc-w1a1 | 6 | 5 cyc, 120.7 k (0.75) | 6 cyc, 101.7 k (0.63) | -16 % |
| tfc-w1a1 | 14 | 14 cyc, 39.2 k (0.24) | 14 cyc, 38.0 k (0.24) | -3 % |
| cybersecurity-mlp | 7 | 6 cyc, 109.5 k (0.68) | 6 cyc, 101.3 k (0.63) | -8 % |
| cnv-w2a2 | 660400 | 460800 cyc, 18.7 k / 188 BRAM (0.39) | 576000 cyc, 19.0 k / 186 BRAM (0.38) | -1 % peak, closer to the target |
| cnv-w2a2 | max | 7200 cyc, 166 BRAM / 22 URAM (0.95) | 7200 cyc, 161 BRAM / 20 URAM (0.94) | -1 % |

(LUT/BRAM from `after3_ZCU104.jsonl`, which reproduces the benchmark's cycles exactly; utilization from the benchmark.) On the
convolutional models the padded search rarely beats the unpadded one: their channel counts
(64/128/256) already have many divisors and a padding DWC costs `in+out` words of cycles. On the
TFC models the benefit is 6-28 % of the LUTs at the same target, and in the "tfc-w1a1 at 14"
case of the task description the padded folding now meets the target (14 cycles, 38.0 k LUT
vs 39.2 k unpadded, vs the baseline's 65.5 k at 10 cycles).

## 5. Search time (ZCU104, effort 50, one SetFolding call, seconds; `before_ZCU104.jsonl` / `after3_ZCU104.jsonl`)

| model | target mode unpadded | target mode padded | maximize unpadded | maximize padded |
|---|---|---|---|---|
| cybersecurity-mlp | 1.0-1.3 -> 1.2-1.7 | 1.2-1.4 -> 2.4-3.1 | 10.4 -> 14.4 | 10.8 -> 28.5 |
| tfc-w1a1 | 1.2-1.4 -> 1.7-2.1 | 1.3-1.4 -> 3.5-4.1 | 12.9 -> 17.5 | 13.7 -> 36.6 |
| tfc-w2a2 | 1.3-1.5 -> 2.0-2.1 | 1.4-1.5 -> 4.0-4.5 | 14.7 -> 22.1 | 15.7 -> 46.0 |
| cnv-w1a1 | 4.2-4.7 -> 6.6-6.8 | 4.2-4.7 -> 12.5-12.9 | 63.4 -> 96.8 | 63.0 -> 190.9 |
| cnv-w2a2 | 4.4-4.7 -> 6.8-6.9 | 4.5-4.7 -> 12.9-13.6 | 63.8 -> 98.2 | 63.8 -> 210.0 |

Unpadded folding takes ~1.5x as long (the whole-graph scoring of every probe plus the
repair/polish pass; the cheapest-seed enumeration is cheap). A padded search is ~3x the old
padded time, which is the price of the guarantee: it contains the full unpadded search (so
padded >= unpadded holds by construction) plus the seeded padded search; in maximize mode the
padded probes replay all unpadded probes and refine on their own. This is within "doubling" for
the unpadded case and beyond it for padding, which is opt-in; `folding_effort` was not changed
(still 50 in builds). The benchmark itself went from 232 s (failing) to ~500 s per board.

## 6. Verification

* `tests/fpgadataflow/test_set_folding.py`: 31 passed, 1 xfailed (pre-existing strict xfail).
  New tests: the optimizer's DWC cycle estimate equals the inserted DWCs', a target is met on the
  graph with DWCs (strictly), padding is never worse (target and maximize mode).
* `test_padding_tightness`: ZCU104 5 passed (499 s), U55C 5 passed (491 s); baseline 5 failed.
* Quick flows (`FEATURE_FLOW_QUICK=1`, folding via the builder + folded cppsim against the golden
  output): 24 passed, 8 skipped -- all 12 `folding_flow` cases (naive and optimizer, 3 models x
  2 boards) and all 10 `padding_flow` cases verify; 8 `padding_flow` cases skip *after* verifying
  because the padded search kept the unpadded folding (no padding DWC to exercise; `dwc pad/all`
  0/N in `feature_flow_report.py`). Every flow's estimated cycles are at or under its target.
* Full flows (HLS synthesis + stitched-IP rtlsim, ZCU104): cnv-w2a2 padded folding with 3
  pad/crop DWCs verifies in cppsim and rtlsim (est. 32400 cycles at target 33025, rtlsim 3086
  fps = the estimate at 10 ns); tfc-w2a2 verifies (8 cycles at target 8) and skips as "no
  padding needed". 3 passed, 1 skipped, 52 min.

## 7. Open questions / notes

* **LabelSelect PE folding** (commit `529dc6c0f`) overrides the original authors' fmax concern.
  It is what the naive folder and the finn-examples build flow do, and the polish keeps PE at 1
  unless the target needs it; still, it changes maximize-mode results (tfc-w1a1 now 5 cycles at
  0.75 instead of 10 at 0.38 on ZCU104; 1 cycle at 0.60 on U55C). Revert = put the tuple back in
  `DO_NOT_FOLD`.
* The padded search rarely beats the unpadded one on the CNNs: with 64/128/256 channels there is
  little to gain, and a padding DWC costs `in+out` words of cycles, which the optimizer now
  prices in. The visible gains are on TFC/MLP (6-28 % LUT at the same target).
* Infeasible targets return the fastest folding the model allows (over budget, with the warning
  and `fits_budget=False`): the cheapest one when several reach the same cycles.
* Padded searches cost ~3x the previous padded time (they contain the unpadded search); unpadded
  ~1.5x. `folding_effort` is unchanged (50). Repair/polish scoring uses the real
  InsertDWC+SpecializeLayers graph (4-35 ms per evaluation on these models); on a MobileNet-size
  graph the polish is bounded by 6 evaluations per folding parameter but was not timed here.
* `tests/benchmark/feature_flow_report.py` crashes (`KeyError: 'flow'`) when
  `padding_tightness.jsonl` is in the results dir (different schema) -- pre-existing, not changed
  (no benchmark-test commits were needed). A `-k "padding_flow or folding_flow"` selection also
  matches the module name `test_dwc_padding_flow.py` and re-runs the tightness tests.
* The optimizer's objective is the budget-normalized weighted sum of LUT/BRAM/URAM/DSP; where the
  naive folder shows fewer LUTs it does so by using nearly all URAM (cnv-w2a2 @99060: 71/77).
