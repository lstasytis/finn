# Plan: FIFO-aware folding

Worktree `/tmp/finn_dev_lstasytis/worktrees/folding-fifo`, branch
`feature/dev/set-folding-optimizer-fifo-aware`, Vivado 2024.2. Logs: `/tmp/finn_dev_lstasytis/agents/folding_fifo/logs/`.

## 0. Baseline (running)

The coordinator's `fold_fifo_cmp.py` (quick mode, `heuristic_analytical` sizing) adapted to this
worktree: naive / optimizer / optimizer+fifo on cnv-w1a1, cnv-w2a2, tfc-w2a2 (ZCU104) ->
`logs/baseline_cmp.log`. Expected: optimizer FIFO kbit several times the naive's, +fifo crashes.

## 1. Repair of the FIFO-aware mode (goal 1)

`insert_and_size_fifos` builds a `DataflowBuildConfig` with `split_large_fifos` (gone) and probes
`TAVGenerationMethod`/`TAVUtilizationMethod` (gone). The analytic sizer is now
`auto_fifo_strategy="heuristic_analytical"`: `InsertDWC -> SpecializeLayers -> GiveUniqueNodeNames
-> AnnotateCycles -> JustInTimeSynthesize(only nodes without tree model) -> DeriveTokenAccessVectors
(tree_model) -> DeriveFIFOSizes -> InsertFIFO -> cleanup_characterization`. Every op in the
benchmark models has a tree model (MVAU, VVAU, SWG, Pool, Thresholding, DWC, LabelSelect, ...),
so this is synthesis-free; a node without one would make JustInTimeSynthesize run vitis_hls.

Design: call the sizer's transformations directly (not `step_set_fifo_depths`, whose
`DeriveTokenAccessVectors` call spawns NUM_DEFAULT_WORKERS processes per call and whose config
needs fields that keep changing), behind a feature probe
`from finn.transformation.fpgadataflow.derive_characteristic import DeriveTokenAccessVectors,
DeriveFIFOSizes` (dev's module has `DeriveCharacteristic` instead). If the probe fails, or any
HW node lacks a tree model: one clear warning, no FIFO term, folding proceeds. The scored copy
gets `num_workers=1` TAV derivation and `cleanup_characterization` (sidecar .npy files).

## 2. Diagnosis of the FIFO growth (goal 2) -- hypotheses, to be measured per edge

The sizer's storage is mostly the **burst floor** (docs: "the rule that sizes most of the
storage"): the consumer's largest demand above the edge's average rate; an SWG reads rows, an
MVAU reads SF words per output window. FIFO *bits* = depth x stream width. Candidate causes:
(a) **stream width**: the optimizer favours large SIMD / parallel_window (wide words, e.g. a
    576-bit SWG->MVAU stream) where the naive folder is capped at mvau_wwidth_max=36 bits of
    weight width and folds PE first; the same depth then costs many times the bits;
(b) **rate mismatch**: the optimizer balances cycles per node to the target but not the
    *shape* of the schedules: a producer that finishes its frame early (fast node before a
    slow one) runs ahead by up to a frame of tokens unless the slack relaxation applies;
(c) **bursty pairings**: an MVAU with small PE / large SIMD consumes its input in one burst per
    output window; behind a row-wise SWG the burst floor grows with the window;
(d) **DWCs**: non-integer-ratio / padding DWCs between differently folded neighbours add edges
    and change the words-per-frame on each side.
Measure: per-FIFO depth, width, kbit, producer/consumer folding for naive vs optimizer
(`scratchpad/fifo_edges.py` on `step_set_fifo_depths.onnx`), on cnv-w1a1 first.

## 3. Candidate fixes (choose by measurement)

A. **Cheap proxy in the cost model**: per edge, estimated FIFO bits = f(burst of the consumer,
   rate ratio, stream width) x width, normalized by the LUT budget like the DWC heuristic.
   Needs to approximate the sizer's burst floor without TAVs: e.g. for an SWG consumer the
   row burst (IFMDim x IFMChannels / SIMD words), for an MVAU SF = MW/SIMD words, times width.
B. **In-loop analytic sizing** (repaired mode): exact but per-candidate cost = TAV derivation +
   chained pass over the whole graph (to measure: ms? seconds?). Could be used in the polish /
   final verdict only (whole-graph scoring, like `_score`), not inside the annealer.
C. **FIFO-aware polish**: extend `_refine` so the polish's cost includes sized FIFO resources
   (B at polish granularity) -- single-step moves that cut FIFO bits at equal cycles.
D. Make the FIFO term default when the sizer exists (`enable_folding_fifo_heuristic` default).
Likely combination: A inside the annealer (cheap, steers away from wide bursty edges) + B/C in
`_score`/polish (exact verdict including FIFO LUT/BRAM), if B is fast enough; report timings.

## 4. Validation

* FIFO kbit / LUT / BRAM / cycles / folding time per model, naive vs optimizer vs
  optimizer+fifo, before/after (quick mode, `heuristic_analytical`).
* `test_padding_tightness` ZCU104 + U55C, `tests/fpgadataflow/test_set_folding.py`,
  `FEATURE_FLOW_QUICK=1 test_folding_flow.py`.
* Full mode on cnv-w1a1 only: optimizer folding with `largefifo_rtlsim` and
  `heuristic_analytical`, throughput unchanged. Check `df -h` first; delete code_gen_*/rtlsim_*/
  vivado_stitch_proj_* in my build dir afterwards.

## 5. Findings so far

* Baseline (old code, quick mode, heuristic_analytical, ZCU104): cnv-w1a1 naive 14.8 kbit vs
  optimizer 36.1 (LUT 98.4 k vs 90.2 k incl. FIFOs 3.5 k vs 6.1 k and DWCs 2.7 k vs 1.4 k, BRAM
  84 vs 86); cnv-w2a2 naive 25.5 vs optimizer 23.3 kbit; tfc-w2a2 7.4 vs 7.8. The gap is a
  cnv-w1a1 thing. `optimizer+fifo` crashed (split_large_fifos).
* Per-edge (cnv-w1a1): 13.4 kbit `DWC -> MVAU_3` (107 x 128 bit), 7.5 kbit `DWC -> MVAU_2`
  (40 x 192), 4.9 kbit `DWC -> MVAU_4` (13 x 384), 2 x 2.2 kbit `SWG(parallel_window) -> DWC` at
  depth 2 (1152-bit words). Root cause: the pairing rule tied SWG SIMD to MVAU SIMD, so every SWG
  runs MH/PE times faster than the MVAU it feeds (SWG_4 146 cycles/frame vs MVAU_3 7200) and
  runs ahead; parallel_window was switched on already at MVAU SIMD == IFMChannels (1152-bit
  words plus a needless DWC). The sizer charges `min(peak run-ahead, consumer demand above the
  frame rate)`; e.g. MVAU_3: 900 words x (1 - 7200/8167) = 107. FIFO LUTs are mostly stream
  width (2-deep 1152-bit FIFO = 1160 LUT), not kbit.
* Repaired in-loop sizer reproduces the build's sizing exactly (36.1 kbit, 37 FIFOs), 0.78 s per
  call on cnv-w1a1, no sidecar files left; the FIFO LUT estimate (proxy) is within ~10 % of the
  sizer on both foldings (5665 vs 6101, 2275 vs 2731).
* New code, cnv-w1a1 at the reference target: 15.5 kbit (1.05x naive), FIFO LUT 2.7 k, DWC
  1.7 k; the search also moved MVAU_4 from (PE 1, SIMD 384: 2.8 k LUT + 22 BRAM + 6 URAM) to
  (PE 4, SIMD 576: 20 k LUT): cheaper by the optimizer's budget-weighted sum (0.805 vs 0.888
  overall) but higher peak utilization (0.64 vs 0.56) -- an objective question (URAM/BRAM
  weight), not a FIFO one; to be reported. Fold time 9.5 s (was 6.6 s).
* In-loop real sizing (`enable_folding_fifo_heuristic=True`): 46.8 s per fold on cnv-w1a1 (the
  polish sizes every candidate) -> stays opt-in; the proxy is the default. Candidates the sizer
  cannot size (a DWC with a non-integer ratio has no tree model) fall back to the proxy.

## 6. Status

- [x] baseline comparison (old code via HEAD's module), per-edge diagnosis
- [x] repair (goal 1): analytic sizer called directly, feature-probed, synthesis-free, declines
      (None + warning) on trees without it or graphs with unmodelled nodes
- [x] fixes: decoupled SWG SIMD (meta value = (mvau_simd, swg_simd, parallel_window)), FIFO
      proxy in the cost model / polish / FoldResult.cost (fifo.sv's cost model when present),
      parallel_window only above the channel count, DWC heuristic covers SWG->MVAU, I/O FIFOs
- [x] unit tests: estimate tracks sizer, FIFO-aware mode, graceful decline
- [x] tightness ZCU104 5 passed (after a fix: with the SWG SIMD free, a padded channel count
      could meet an SWG SIMD the DWC cannot pad within a word -> such SIMDs are excluded).
      Unit tests 34 passed / 1 xfailed. Stray `tav_*` dirs: the work-dir attribute was set on
      stale node objects -> fixed, 7860 dirs removed.
- [x] commits ec9929b36 (FIFO-economical foldings), e30a3f6ba (FIFO-aware mode repaired),
      1f9713080 (tests)
- [x] U55C tightness 5 passed; quick folding flows 12 passed; final comparison (table in REPORT.md);
      full-mode cnv-w1a1 verified with both sizings at 12255 fps (15.5 / 27.6 kbit)
- [x] REPORT.md, build-dir cleanup
