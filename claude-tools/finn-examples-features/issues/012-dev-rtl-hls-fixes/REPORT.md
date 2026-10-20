# Issue 012 report: two dev fixes as separate PR branches
(Saved by the orchestrator from the agent's hand-back; PR texts: PR_feature-dev-mvu-output-queue.md,
PR_feature-dev-split-concat-dataflow.md; scripts in scripts/.)

## feature/dev/mvu-output-queue (55a0edb83, one commit on upstream/dev; worktree build/wt/pr-mvuq)
`mvu_vvu_axi.sv`: the output-queue depth uses mvu.sv's latency formula, `MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1`,
and a module-level `USE_8SX9_DSP58` drives both the core selection and the depth (the three lane localparams moved
out of `blkDsp` unchanged). Third change beyond the two approved lines: the depth formula keyed on `VERSION == 3`,
but non-INT8 configurations on DSP58 (e.g. 4x4 bit) use the deeper mvu.sv core; with only the two lines the Versal
4-bit case still loses words. The open DSP58 value check was the same overflow (signed queue pointer `OPtr` wrapped
to "empty", whole words dropped).
Standalone reproducer (xsim, 50k cycles each): dev DSP58 INT8 SIMD 24: 4992 of 4994 outputs wrong; dev DSP58 4x4
SIMD 32: 999 X words; two lines only, same 4x4 config: 608 X words; full fix, 14 configurations (SIMD 1/3/64/75/96,
DSP48E1/E2, DSP58 INT8 and generic core, behavioural and structural, SF 2, the recorded espcn pattern): 0 X, 0
wrong, queue exactly full at worst. dev's own `mvu_axi_tb.sv` passes even on dev (its output side never stalls long).
FINN test `test_fpgadataflow_rtl_mvau_output_backpressure` (MVAU_rtl SIMD = MW = 24 feeding an MVAU_hls that reads 32
words then stalls 480 cycles; zu7ev and vc1902, 4- and 8-bit): dev 0/4, two lines only 3/4, full fix 4/4.
Regression `test_fpgadataflow_rtl_mvau` (144 selected): 72 passed, 50 skipped by the test, 22 failed = xcku3p cases
("Specified part could not be found", same on dev).
espcn end to end (worktree build/wt/fix012-espcn, branch fix/012-espcn = test2/fifo + this fix; designs re-stitched
with unchanged IPs and FIFO depths, full frame): committed folding + analytic FIFOs 20460 zero outputs -> 0 wrong
(999,901 cycles/frame); optimizer folding at the committed throughput 31920 zero outputs -> 0 wrong (1,200,526).
Both match the Python HW model exactly and the golden within 1/255 (12 known rounding spots), bit-identical to the
passing fold-max build's output.

## feature/dev/split-concat-dataflow (bcb838bbb on upstream/dev; worktree build/wt/pr-splitdf)
On dev `get_exp_cycles` is already 1 word/cycle: only the `#pragma HLS dataflow disable_start_propagation` top and
tests. csynth: latency 4 / interval 5 / not pipelined -> latency 2 / interval 1 / dataflow; LUT 198-264 -> 115-143,
FF 39-177 -> 26-53. New tests 6 passed with the pragma (fail at exactly 5x without: 640 vs 128 ...); existing
split/concat tests 6 passed. Conflicts in intent with the x5 estimate commit 143b54492 (issue 010) on
feature/dev/hls-exp-cycles-overhead: if this PR is taken, that commit is dropped (the orchestrator did so; the commit
stays on fix/010-dev).

## Housekeeping
Deleted: scratch worktree build/wt/pr-mvuq-dev, build dirs fx/{pr-mvuq, pr-mvuq-dev, pr-splitdf, fix012-espcn}
(~2.2 GB). Kept: worktrees pr-mvuq, pr-splitdf, fix012-espcn, scratchpad i012/. Push commands (for the user):
`git push origin feature/dev/mvu-output-queue feature/dev/split-concat-dataflow`.
