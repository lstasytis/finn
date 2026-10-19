# Report 006: balanced (optimizer) foldings 2-5 % below the estimate — FIFO sizing, fixed
(Written by the orchestrator from the fix agent's hand-back. Scripts in `scripts/`, data in `data/`.)

## Answers
1. **FIFO sizing.** Re-stitched with every FIFO at max(depth, 8192) (diagnostic): interval = estimate
   (cnv-w2a2 AUP-ZU3 28226 -> 27655; cnv-w1a2 KV260 13262 -> 12551). The slack relaxation in
   `edge_depth` gave up run-ahead that a consumer driving the pacer needs.
2. No node is slower than `get_exp_cycles()` (each stitched alone, 4 frames: MVAU_hls exact;
   SWG_rtl, DWC_rtl at or below).
3. Not a measurement artefact (2, 4, 8 frames identical).

## Root cause
* **w2a2, pacer-bound chain:** pacers MVAU_hls_2/4 (27655), MVAU_hls_3 25607. Edge
  `DWC_rtl_4 -> MVAU_hls_3`: the sizer reset the chain period after a pacer (chain counted as 14690
  cycles), allowance 187 > peak 48, burst floor 32 set the depth. Below a pacer the chain runs at the
  global period and can't catch up once blocked (SWG_rtl_4 waits ~5000 cycles per frame for 26 new
  pixels, more than MVAU_hls_3's 2048 slack). FIFO_15 depth 32 -> 28226, 48 -> 27655.
* **w1a2, producer pauses at frame boundaries:** pacer MVAU_hls_1 (12551), chain above MVAU_hls_0
  (10807). Edge `DWC_rtl_3 -> MVAU_hls_1`: peak 404, allowance 436 -> depth 4. At every frame boundary
  SWG_rtl_1 needs 2 rows + 3 pixels before its first window (824 cycles). Depth 4 -> 13262, 100 ->
  12881, >= 184 -> 12551.

## Fix (FIFO branch; `derive_characteristic.py`, docs, unit test)
Only for consumers that drive the pacer:
* **Pacer-bound chains:** allowance uses `t_full` (slowest period of the whole upstream chain, no
  reset after a pacer): no relaxation below a pacer.
* **Refill floor:** depth >= min(peak, tokens the consumer reads during the producer's longest write
  gap), not counting the first `consumer slack + upstream_cover` cycles of the gap (`upstream_cover`
  = sum of depth*global_period/per_frame of the FIFOs above the producer back to a pacer, join or
  graph input; without it committed mobilenet_v1 over-sized `MVAU_hls_1 -> DWC_rtl_4` 2 -> 677).
Commit `f1cd9abc3` on `fix/006` (on test/dwc), `ba7cfe45a` on `fix/006-pr` (on the FIFO PR branch;
`test_fifosizing_chained_tav.py` 3 passed). Files: `derive_characteristic.py`,
`docs/finn/fifo_sizing_heuristics.rst` (sections "Pacer-bound chains", "Refill floor"),
`tests/fpgadataflow/test_fifosizing_chained_tav.py`. Lint clean.
Orchestrator: fast-forwarded `feature/dev/analytical-fifo-sizing` to `ba7cfe45a`, cherry-picked onto
`test/{fifo,folding,dwc}` and the integration branch (18:05), requeued the affected quick jobs.

## Evidence (fix/006, FINN_BUILD_DIR fx/fix006)
| test | result | rtlsim/est | kbit |
|---|---|---|---|
| bnnpynq_folding[AUP-ZU3_8GB-cnv-w2a2-maximize] | PASSED | 1.000 (27655) | 99.4 (was 88.9) |
| bnnpynq_folding[KV260_SOM-cnv-w1a2-maximize] | PASSED | 1.000 (18439) | 50.2* |
| bnnpynq_fifo_sizing[AUP-ZU3_8GB-cnv-w1a1-heuristic_analytical] | PASSED | 1.0009 | 11.9 (unchanged) |
\* ran before the upstream-cover refinement (one edge 56 vs final 50; re-simulated 50: 18439). The
current optimizer gives KV260 a different folding (18439) than the issue's build (12551), on which the
old sizing already reaches the estimate: here the fix costs +9.4 kbit (37.5 -> 46.9) for no gain.
The issue's KV260 build re-simulated with the new depth 184: 12551 (= estimate).

Offline re-sizing of 62 campaign builds (`data/eval*.log`): 28 change. gtsrb maximize 12992 -> 12806
(est 12807), cnv-w1a1 maximize 14457 -> 14406 (est 14407), cnv-w2a2 target 33320 -> 33206. Committed
foldings: only vgg10-radioml changes (four FIFOs 2 -> 3, 14.2 -> 15.7 kbit); cnv, gtsrb, kws, tfc,
lfc, mobilenet_v1 unchanged.

**Caution:** two HLS IP top files of the issue's KV260 build in `fx/folding` (MVAU_hls_6.v,
MVAU_hls_8.v + one sub-file each) were found 0 bytes (rewritten 14:25/14:38, writer not identified,
possibly the agent's own re-stitch of that build). Don't reuse that build without regenerating them.

## mobilenet_v1 U55C maximize (0.848)
No node slower than its estimate; ~80 nodes at 50176-50183 cycles (VVAU_hls has a ~+8..10 per-frame
HLS overhead but is far from the bottleneck). With the fix's depths 59171 -> 56832 cycles
(0.848 -> 0.883) for +491 kbit (642 -> 1133); all FIFOs 8192 also 56832: the remaining 13 % is not
FIFO-related. Hypothesis (unverified): with ~80 nodes at the target period and zero slack, per-frame
dead times add up -> the optimizer should leave margin. A folding question.

## mobilenet_v1 U55C DWC test (0.9756)
All FIFOs 16384 -> interval 788552 = estimate (was 808289): FIFO sizing, but the new rules don't
change this build (740.0 kbit, same depths). Padding DWCs have no tree model but are characterized by
rtlsim via JustInTimeSynthesize (real TAVs). The under-sized edge is not identified (~1.5 h per sim).

## Open points
* Both rules are conservative (ignore in-node buffering): KV260 +25 % FIFO storage with no gain,
  mobilenet maximize +76 % for +4 %. Exact sizing needs a bounded-FIFO simulation (out of scope).
  Cheaper idea: below a pacer cap the allowance by the upstream cover instead of zeroing it.
* A cheaper place for the refill buffer is in front of the refilling SWG (narrow stream), e.g.
  cnv-w1a2 KV260 63 x 128 bit instead of 184 x 288 bit; not implemented.
* Not run: `test_fifosizing_linear` (slow) and the rest of the FIFO suite.
Worktrees: `build/wt/fix006` (fix/006), `fix006-pr`, `fix006-diag`, `fix006-base`.
