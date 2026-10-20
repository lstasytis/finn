# Issue 010 report: transformer_benchmark maximize folding, rtlsim 0.2x the estimate
Two agents. The first found the Split/Concat cause and committed `fix/010`, `fix/010-dev`, `fix/010-folding`
(stopped by a usage limit, no report). The second (this report, saved by the orchestrator) reviewed and verified
that, added a test, did the FIFO-sizing part (incl. issue 009's 0.965x) and ran the feature tests.

## Summary
| build (ZCU104, transformer_benchmark) | est cycles | rtlsim cycles | ratio |
|---|---|---|---|
| maximize, test2/folding (the issue) | 2048 | 10240 | 0.20 |
| maximize after Split/Concat fixes (fx/fix010/fold-max_..._5wc25vjv) | 1280 | 1755 | 0.73 |
| same build, all FIFOs 4096 | 1280 | 1280 | 1.00 |
| same build, only StreamingFIFO_rtl_9 (residual) at 4096 | 1280 | 1280 | 1.00 |
| maximize, fix/010 all fixes, feature test (fold-max_..._v0z1pmrn) | 1280 | 1755 | 0.73 (FAILED, FIFO_9=1232) |
| committed folding, fifo_sizing[heuristic_analytical] (fifo-..._2e75dh0n) | 65536 | 65536 | 1.00 (was 0.965) |
| optimizer at committed target, folding[target] (fold-target_..._2czmbmx6) | 65536 | 65536 | 1.00 (was 0.965) |

Root causes: (1) StreamingSplit_hls/StreamingConcat_hls take 5 cycles/word (unpipelined HLS top around a
single-word hlslib function; csynth Latency 4, Interval 5), get_exp_cycles said 1; the optimizer never folded their
SIMD (DO_NOT_FOLD). (2) The analytical FIFO sizer leaves residual shortcuts short: TAVs record the steady state at an
arbitrary phase; an MVAU_rtl TAV opens with a feature map's NF outputs whose read burst wrapped to the window end, so
each MVAU_rtl in the long branch looks one feature map faster. Fixed for the committed folding; at maximize the
shortcut is still short because TAVs carry no block latency (InnerShuffle page fill, dynamic MVAU matrix load) and
no pipeline latency.

## Per-node rtlsim vs estimate
Single-node stitched-IP rtlsim, 4 frames, steady state (scripts/nodesim.py). Full tables:
data/rtlsim_vs_estimate.md (issue's maximize folding j33hxjs7, 77 nodes), data/rtlsim_vs_estimate_maximize.md
(5wc25vjv, 80 nodes, with the sizer model's first-frame latency).
Issue folding (Split/Concat SIMD 1): StreamingSplit_hls est 2048 rtl 10240 (5.000); StreamingConcat_hls 10240/10240
(fixed estimate); InnerShuffle_rtl 1536/1024 and 384/256 (0.667, 6 nodes); HWSoftmax_hls 1214/1152 (0.949);
MVAU_rtl incl. dynamic 2048/2048 (12); Thresholding_rtl 2048 and 1024 exact (16); DWC 2048/512 exact (33);
ElementwiseMul/Add, DuplicateStreams exact.
After fixes (Split/Concat SIMD 8): Split 1280/1280 (3), Concat 1280/1280, InnerShuffle 768/512 (0.667, 6),
HWSoftmax 1214/1152 (0.949, 4), MVAU_rtl 1024/1024 (12), Thresholding 1024 (16), DWC 1024/256 (29),
Elementwise/Dup exact. No node slower than its estimate; full graph with deep FIFOs = 1280 exactly. Not fixed
(harmless over-estimates): InnerShuffle get_exp_cycles = 2N + page includes the first-page fill (steady 2N);
HWSoftmax 5 % pessimistic.

## Fix 1: Split/Concat (first agent, verified)
* 143b54492 (fix/010-dev, off feature/dev/hls-exp-cycles-overhead) = 90503f64c (fix/010): get_exp_cycles x5;
  tests test_fpgadataflow_{split,concat}_exp_cycles_rtlsim[simd 1,2,4]: 6 passed. The FIFO branch's tree models
  already use Interval 5.
* 9cb25debf (fix/010-folding, off feature/dev/set-folding-optimizer) = ac74a3e97: the optimizer folds Split/Concat
  SIMD over common divisors of ChannelsPerStream. Reviewed OK. Added test a948f2979 (= 65fb66aac on fix/010): both
  styles fold a Split/Concat pair to SIMD | 8 and meet 1/4 of the unfolded cycles; fails without the fix.
  test_set_folding.py: 41 passed, 5 skipped, 1 xfailed.
* Option, not committed: `#pragma HLS dataflow disable_start_propagation` in the generated top (as
  DuplicateStreams_hls has) makes both 1 word/cycle (branch fix/010-dev-dataflow, worktree build/wt/fix010-df; the
  same 6 tests with 1x expectation pass). HLS codegen change, excluded by the rules: user decision.

## Fix 2: residual shortcut FIFOs
Committed folding (fx/folding2/fold-target_..._e4ijhrzw): Thresholding_rtl_1 -> ElementwiseAdd_hls_0 sized 1985.
Offline on the build's TAVs (tav_model.py, size_offline.py, edge_sched.py): the join reads the first 63 tokens of
frame 1 with zero lag, then pauses 63187 cycles; the steady frame lags 64070 for every token. The 63 early tokens
come from MVAU_rtl TAVs: mvau_rtl_tree rotates the window (shift = 2e), feature map 0's read burst wraps to the
window end and the TAV opens with its NF outputs before any read (31 for MW=MH=32 SIMD=PE=1; all 12 MVAU_rtl,
tav_lead.py); two on the critical path -> 2x31+1 = 63. causal_writes only applies to compacting nodes at token
granularity; the join edge uses the fill frame alone (fill_only).
Measured need (full-graph rtlsim, only that FIFO changed; fullsim.sh, data/fullgraph_fifo_sweep.txt):
1985/2003/2020 -> 67895; 2048/2064 -> 67415; 2077..4096 -> 65536.
Fix c74d17dc5 (fix/010-fifo, off feature/dev/analytical-fifo-sizing) = 8ec4e2f84 (fix/010): "FIFO sizing: renumber
output tokens past a TAV's phase lead": in the causal pass, output tokens written before the first read on a
streamed input (phase lead) are the previous frame's; outputs renumbered past them (all frames). Not applied to
nodes writing more words per frame than they read (FMPadding writes padding first: leads 693-2176 on resnet50;
applying it there doubled the shortcuts, +64 % depth). Docs: new "Phase lead" rule in fifo_sizing_heuristics.rst.
Effect (offline): committed folding FIFO_9 1985 -> 2077 (needs 2065-2077: exactly at the threshold); maximize FIFO_9
1212 -> 1232 (needs 1501-1550); resnet50 V80 total 362645 unchanged. Limitation: lead detected from the TAV only, so
expanding MVAUs (dynamic Q K^T; SIMD wider than MH/PE) are not corrected.
Tests: new test_chained_tav_residual_mvau_rtl (fork -> 2x MVAU_rtl SIMD=PE=1 -> join; shortcut 27 -> 88, same as with
MVAU_hls whose tree has no lead; 27 with phase_lead disabled) + phase_lead/renumbering checks in
test_periodic_peak_and_floors; test_fifosizing_chained_tav.py 4 passed. Feature tests on fix/010:
fifo_sizing[...heuristic_analytical] PASSED 1.00x, FIFO 89.7 kbit (was 89.0); folding[target] PASSED 1.00x.

## Maximize: what is left
The short residual shortcut is the whole remaining gap: at 5wc25vjv all FIFOs 4096 -> 1280; only FIFO_9 (or FIFO_2,
the same shortcut before Thresholding_rtl_1) at 4096 -> 1280; enlarging every other FIFO group instead -> 1755-1785.
FIFO_9 sweep: 1212/1232/1237/1260 -> 1755; 1300..1500 -> 1435; 1550..4096 -> 1280. The fix adds 20 tokens; ~300
more are needed. Node first-frame latency, rtlsim vs sizer model (tav_latency.py): InnerShuffle +263 cycles (a page
must be in the bank before any word leaves; tree writes start at 0), attention MVAUs +69, Thresholding +18,
projection MVAUs +22, Split +16, ElementwiseMul +11; the dynamic Q K^T MVAU emits from cycle 252 in the model while
its weights complete at 1548. Negligible at 65536 cycles/frame; at 1280 (1.25 cycles/token) ~400 cycles = ~300
tokens. Not fixed. Tried (patches in scripts/): experimental_block_causality.patch (ops declare block-consumed
inputs via get_tav_input_blocks; causal pass holds output blocks): committed folding 2112, maximize 1269, resnet50
+0.3 % depth; still short because pipeline latencies are in no TAV. experimental_join_steady_frame.patch (join
inputs take max(fill, steady)): 2003, not enough. A complete fix needs per-node latency in the TAV model (block hook
+ per-op pipeline latency would give ~1650 by these numbers): a design change to the FIFO PR, stopped here.

## Which commit goes on which branch
143b54492 -> fix/010-dev (base feature/dev/hls-exp-cycles-overhead 181c36d58): Split/Concat x5 + rtlsim tests.
9cb25debf, a948f2979 -> fix/010-folding (base feature/dev/set-folding-optimizer 57594bf0c): optimizer Split/Concat
SIMD + test.
c74d17dc5 -> fix/010-fifo (base feature/dev/analytical-fifo-sizing e8c2cc992): phase lead + tests + docs.
90503f64c, ac74a3e97, 65fb66aac, 8ec4e2f84 -> fix/010 (base test2/folding): the above, for the feature tests.
Uncommitted: fix/010-dev-dataflow (build/wt/fix010-df): HLS pragma alternative.

## Tests run
fix/010-dev: split/concat exp_cycles rtlsim 6 passed (with the pragma: 6 passed). fix/010-folding:
test_set_folding.py 41 passed/5 skipped/1 xfailed. fix/010-fifo: test_fifosizing_chained_tav.py 4 passed. fix/010:
fifo_sizing[ZCU104-transformer_benchmark-heuristic_analytical] PASSED, folding[...-target] PASSED,
folding[...-maximize] FAILED (0.729 < 0.98, FIFO_9 1232). 34 full-graph FIFO sweeps; 80-node per-node sweep.

## Aside (not fixed)
A fork fed directly by the graph input is skipped in derive_chained_tav_depths (Step 5a continues when no input edge
is known): its shortcut is never sized and the branch's first node gets no input schedule.

## Files and deletions
Scripts: tav_model.py, size_offline.py, edge_sched.py, tav_lead.py, tav_latency.py, tav_phase.py, fullsim.sh,
run_sweep_max.sh, two patches; first agent's nodesim.py, run_sweep.sh, fold_max_est.py, size_dbg.py, table.py. Data:
rtlsim_vs_estimate*.md, fullgraph_fifo_sweep.txt, nodesim_results.txt. Kept: fx/fix010/fifo010/{target,max}_tav.onnx.
Deleted in fx/fix010 (~3.1 GB): rtlsim_finn_design_wrapper_*, vivado_stitch_proj_*, code_gen_ipgen_StreamingFIFO_*,
tav_*, dataflow_partition_*, t_fold, t_fifo, t_fix010-dev, t_fix010-df, szdbg_target, hlsexp, two test_fifo_* dirs,
fold-max-est_..._smshr7gk, fifo010/r50_tav.onnx, scripts/__pycache__, toy_residual_sim.py.
