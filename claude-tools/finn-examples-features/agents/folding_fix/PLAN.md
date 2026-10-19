# Plan: reliable folding optimizer, padding a strict win

Worktree `/tmp/finn_dev_lstasytis/worktrees/folding-fix`, branch
`feature/dev/set-folding-optimizer-fixes`. Logs/results: `/tmp/finn_dev_lstasytis/agents/folding_fix/logs`.

## 0. Baseline (done)

`pytest tests/benchmark/test_dwc_padding_flow.py -k "tightness and ZCU104" -n 4`: 5 failed in
232 s; the JSONL is byte-identical to the supplied baseline (seed 0 => deterministic, so
before/after comparisons are exact). Timed per-call sweep (optimizer unpadded/padded + naive)
running into `logs/before_ZCU104.jsonl` (TFC: ~1.5 s per SetFolding call at effort 50).

## 1. Diagnosis

A. **DWC cycles are invisible to the optimizer** (confirmed on tfc-w2a2, target 56): the
   optimizer's own graph sits at exactly 56 cycles, but the benchmark inserts DWCs first.
   `StreamingDataWidthConverter.get_exp_cycles` is `in_words + out_words` for non-integer
   width ratios and for padding/cropping, so the 28->32 bit DWC between Thresholding (PE=14,
   56 words) and MVAU_0 (SIMD=16, 49 words) costs 105 cycles. The cost model only takes
   `max(node cycles)` over the partition and `_fold_at` scores `achieved` without DWCs.
   This hits padded runs harder: every pad/crop DWC is "sum of both sides", i.e. up to 2x
   the neighbours' cycles, which is why padded runs miss targets the unpadded ones meet
   (tfc-w1a1 6 -> 13, 14 -> 17) and why padded maximize is slower (cnv-w1a1 5184 -> 7119).
B. **Hard constraint relaxed by 2 %**: `value_to_minimize_relaxation=0.98` makes
   `cycles <= 1.0204 * target` count as feasible (cnv-w1a1 57169 -> 57600).
C. **Budget verdict without DWCs**: `fits` in `_fold_at` aggregates node resources only, so
   padded foldings whose pad/crop DWCs cost LUTs end up over budget (cnv-w2a2 maximize,
   utilization 1.007 after DWC insertion).
D. **Search noise, no anchoring**: the padded run is an independent SA in a larger space with
   the same evaluation budget per parameter; nothing ties it to the unpadded solution, so it
   can return something worse although its space is a superset. Partitions of 3 nodes are
   optimized left to right, each seeing its own nodes' cycles only.
E. **Superset not guaranteed by construction**: `_pair_swg_with_mvau` / `_pair_swg_with_vvau`
   iterate `set(ifm_values)` and de-duplicate by `mvau_simd` / folding factor, so a padded
   (simd, MW) could shadow the unpadded one if the set iterated the padded value first (small
   ints iterate ascending in CPython so it does not bite today, but it is unordered).
F. **No final check/repair against the target** on the whole graph, and no "best feasible
   seen" across padded/unpadded; maximize mode's bisection can follow different paths for
   padded vs unpadded.

## 2. Fixes (all in set_folding.py)

1. DWC-aware cycles: `Optimizer._dwc_cycles(producer, consumer)` mirroring
   `StreamingDataWidthConverter.get_exp_cycles` + InsertDWC's insertion rule; `cost_model`
   takes `max(node cycles, DWC cycles to the producer)` per partition node.
2. `value_to_minimize_relaxation` -> 1.0 (a target is a hard target).
3. Score every probe on the as-inserted graph (InsertDWC + SpecializeLayers on a copy):
   `achieved` and `fits` include DWC cycles and resources, exactly what the build carries.
4. Seeded padded search: `Optimizer.optimize(initial_guess=<meta values>)`; with padding > 0
   `_fold_at` runs the unpadded SA first, then the padded SA seeded with it, and keeps the
   better (meets target & fits first, then cost / cycles). Deterministic seed => the unpadded
   sub-run equals the standalone unpadded run, so padded >= unpadded per probe by construction.
5. Repair + polish on the full graph (coordinate descent over meta values, DWC-aware):
   repair = raise the folding of the bottleneck node/DWC neighbours until the target is met
   (terminates: indices only increase; max folding has no slow DWCs); polish = single-step
   moves that keep the target and lower cost. Runs after the SA in every probe.
6. Maximize mode with padding: unpadded search, padded search, then one seeded padded probe at
   the unpadded best's cycles; return the best. Guarantees padded <= unpadded.
7. `sorted(set(ifm_values))` in the pairing rules (unpadded values claim their factors first).
8. Check whether dual_annealing's local search wastes evaluations on the step objective
   (`no_local_search`); adopt only if measured.

## 3. Validation

* `test_padding_tightness` ZCU104 + U55C; also compare against `style="naive"` for goal 1.
* `tests/fpgadataflow/test_set_folding.py`.
* `FEATURE_FLOW_QUICK=1` padding/folding flows (cppsim), then full flows for tfc-w2a2 and
  cnv-w2a2 (HLS + rtlsim).
* Timing: before/after sweep per model/target (effort 50).
* Lint: isort/black/ruff from the manual's venv.

## 4. Status

- [x] baseline reproduced, failure modes A-C confirmed by per-node diagnostics
- [x] remaining diagnostics: tfc-w1a1 6 padded misses via a 28->7 bit cropping DWC (A);
      tfc-w1a1 14 unpadded = plain SA under-convergence (65 k LUT at 8-10 cycles while a
      14-cycle folding with half the MACs exists) -> fix 9 below
- [x] fixes 1-7 implemented (round 1): ZCU104 benchmark 5 passed, 607 s (baseline 232 s)
- [x] fix 8 measured: local search wastes 10-15 % of the evaluations, results identical ->
      not changed
- [x] fix 9 (new): `initial_guess="cheapest"` -- per node, enumerate its own PE/SIMD
      combinations and start the annealer from the cheapest one meeting the target; the
      annealer keeps the start unless it finds better, so the result is never worse than
      that node-wise optimum. tfc-w1a1@14: 65.5 k (10 cyc) -> 39.2 k LUT (14 cyc) unpadded.
      Surprise: calling it for the resource-type pass too left PE/SIMD at the last
      enumerated combination (that pass only re-applies ram_style/resType) -> restricted
      to the folding pass.
- [x] round-2 ZCU104 benchmark: 5 passed, 562 s (baseline 232 s, failing). Padding gains:
      tfc-w2a2 @12 util 0.75->0.53, @24 0.39->0.28, max 0.75->0.65; tfc-w1a1 @14 0.24->0.23;
      cybersecurity @7 0.68->0.63. Infeasible targets (tfc-w1a1 2/3/6 < fastest 10,
      cybersecurity 1/3) now return the cheapest fastest folding (tfc-w1a1: util 0.40 in
      budget instead of 1.49 over budget).
- [x] unit tests: 31 passed, 1 xfailed (pre-existing)
- [x] commits 0173d242d (optimizer fixes), 9da4e5304 (cheapest seed), be14915e6 (tests)
- [x] timed after-sweep (`logs/after_ZCU104.jsonl`): total SetFolding time over the 30
      (model, target) cases unpadded 228 -> 366 s, padded 231 -> 781 s (padded now contains
      the unpadded run). U55C benchmark: 5 passed (572 s).
- [x] naive comparison: naive met tfc-w1a1 targets 2/3/6 (1/2/5 cycles, 6 within budget) by
      folding LabelSelect PE, which the optimizer kept at 1 (DO_NOT_FOLD) -> fix 10: fold
      LabelSelect PE over `Labels` (PE_MAX_OVERRIDE; the output width is only top-K) and let
      the polish take less-folding steps at equal cost so a zero-cost parameter is folded only
      as far as needed. tfc-w1a1@6: 5 cycles / 0.75 (= naive), padded 6 / 0.63; @40 LabelSelect
      stays PE=1. Commit 529dc6c0f. "Naive cheaper" rows on CNV are LUT-only artefacts: e.g.
      cnv-w2a2@99060 naive 24 k LUT but 71/77 URAM (peak util 0.92) vs optimizer 59 k LUT /
      peak 0.37; the optimizer's weighted sum and the benchmark's peak utilization both favour
      the optimizer.
- [x] final (fix3) benchmarks: ZCU104 5 passed (499 s), U55C 5 passed (491 s); tfc-w2a2@8 now
      met too (LabelSelect). Timing sweep after3 done (matches the benchmark's cycles exactly).
      Unit tests 31 passed / 1 xfailed.
- [x] quick flows (cppsim via the builder): 24 passed, 8 skipped (padding_flow cases whose
      folding needed no padding DWC; they verified before skipping); cnv-w2a2 padded flows pass
      on both boards.
- [x] full flows (HLS + stitched-IP rtlsim) ZCU104: cnv-w2a2 padded (3 pad/crop DWCs) verifies,
      tfc-w2a2 verifies and skips (no padding needed); 3 passed, 1 skipped, 52 min
- [x] REPORT.md complete
- [ ] REPORT.md
