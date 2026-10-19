# Report 001: bert (VCK190) heuristic_analytical FIFO sizing — fixed
(Written by the orchestrator from the fix agent's hand-back; subagents cannot write report files.)

## Root cause
`MVAU_rtl_3` (and `MVAU_rtl_4`), the two `mem_mode="dynamic"` attention matmuls (Q*K^T, softmax*V):
1. `MVAU.get_tree_model()` declined `dynamic` (and `external_mem`), so `heuristic_analytical` fell
   back to rtlsim characterization for them.
2. `MVAU.derive_token_access_vectors()` drove `in1` only for `internal_decoupled`/`external`; for
   `dynamic` only `in0`. The node-level rtlsim is the bare MVU (`dynamic_load` exists only in the
   stitched IP), so the MVU waited for weight tiles forever -> `out0_V` watchdog after 98304 cycles.
Hidden second bug: the TAV had a single row (`in0`), so `stream_curves` gave `in0`'s curve to the
`in1` edge too (MVAU_rtl_4: in0 24576 words/frame, in1 6144) -> wrong token count on the
reconvergent V-branch edge. `HWSoftmax_rtl_0` (the only other node without tree model)
characterizes fine (period 49173 vs estimate 49234).

## Fix (`fix/001`, worktree `build/wt/fix001`; PR-branch copies on `fix/001-pr`)
* `1c7d93f2e` (PR) / `d084b09b4`: `matrixvectoractivation.py`
  - rtlsim fallback: `dynamic` drives `in1` like `internal_decoupled` (one SIMDxPE tile per compute
    cycle, `prod(numInputVectors) * calc_wmem()` words).
  - `get_tree_model()` accepts `dynamic` (still declines `external_mem`, and `dynamic` when
    `numInputVectors[-1] < SIMD`, where `dynamic_load` cannot refill one bank while replaying the
    other). Tree vs rtlsim on bert's dynamic MVAUs: in0/out0 within 3 tokens, period 98306 vs 98305;
    scaled-down shapes agree for RTL and HLS.
  - New `MVAU.dynamic_in1_tav()`: appends an `in1` row (one burst of `MW*MH/PE` words per matrix,
    starting at the first `in0` read of that matrix, modulo the period) after either
    characterization path; `stream_curves` now binds each streamed input to its own curve.
* `19ed61ff7` (PR) / `71023c61e`: tests in `tests/fpgadataflow/test_fpgadataflow_mvau.py`
  (decline test uses SIMD=32 vs 18 vectors; new `test_mvau_dynamic_tav_streams` (no Vivado); new
  `test_fpgadataflow_analytical_characterization_mvau_dynamic` (tree vs rtlsim, rtl+hls, 3 foldings)).
* Orchestrator: fast-forwarded `feature/dev/analytical-fifo-sizing` to `fix/001-pr` and
  cherry-picked both commits onto `test/{fifo,folding,dwc}` and `dev-finnexamples-features`.

## Evidence
`test_bert_fifo_sizing[VCK190-heuristic_analytical]` PASSED (48 min), build
`build/finn_build/fx/fix001/fifo-heuristic_analytical_bert_VCK190_3wk9udu5/`: output correct;
rtlsim interval 98304 cycles = 2034.5 fps vs the build's estimate 508.6 fps (ratio 4.0); latency
156661 cycles; FIFOs 6085 kbit in 90 FIFOs (largest: 21010 residual DuplicateStreams_hls_0 ->
ElementwiseAdd_rtl_4; 4604 Q -> MVAU_rtl_3; 4596 V -> MVAU_rtl_4.in1; 4096 K -> MVAU_rtl_3.in1);
`step_set_fifo_depths` 20 s. `test_fifosizing*.py -m "not slow"` 4 passed; slow `multi_io`,
`residual_matches_gauge_maxfill`, `linear` tfc/cnv (heuristic_analytical, heuristic_rtlsim) passed.
Rule breach: `linear[tfc-largefifo_rtlsim]` (part of that unit test's parametrization) ran once by
mistake (passed); the cnv one was then excluded. New MVAU tests pass on both branches. Lint clean.

## Open points
* bert's estimate is 4x pessimistic: `get_exp_cycles` multiplies by TH for tiled MVAUs (TH=8 for
  MVAU_rtl_0/1/2/6/7: 393216 vs tree period 49152); the real interval 98304 is set by the dynamic
  MVAUs. The 0.98x check stays valid if that is fixed (98304 vs 98304).
* Load-bound dynamic MVAUs (`numInputVectors[-1] < SIMD`) only approximated (tree declines, rtlsim
  fallback ignores the dynamic_load bound; counts exact, timing optimistic). bert doesn't hit it.
* `heuristic_rtlsim` for bert not run (scheduled at the end of the campaign).
