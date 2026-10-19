# Report 002: tfc rtlsim throughput 0.53x the estimate — root-caused and fixed
(Written by the orchestrator from the fix agent's hand-back.)

## Root cause
The HLS top functions of `MVAU_hls` and `LabelSelect_hls` are not pipelined (`ap_ctrl_none`, one
invocation per frame): consecutive frames don't overlap, so every frame also pays the fill/drain of
the inner II=1 pipeline. `get_exp_cycles()` counts only the trip count. Vitis csynth agrees
(MVAU_hls_1: top latency 15, interval 16, not pipelined; inner trip 8, depth 4).
1. The 15 cycles come from MVAU_hls_0..2, not from FIFOs: with all FIFOs hand-set to 256 and
   re-stitched (diagnostic only) the interval stays 15. Nodes stitched alone (16 frames):
   Thresholding_rtl and DWCs match their estimates; MVAU_hls_0 (SIMD 392, PE 16) 8 -> 15,
   MVAU_hls_1/2 8 -> 15, MVAU_hls_3 5 -> 11, LabelSelect_hls_0 6 -> 9.
2. Not a measurement artefact: 2, 4, 16, 64 frames all give 15 cycles.
3. Cause is dev's estimate; FIFO sizing and DWC cycles are correct.
Standalone sweep: MVAU_hls +5..+7 over 21 configs (trip 1-256, SIMD 8-512, INT2-INT8, binary,
with/without thresholds), one INT8-threshold config +8; LabelSelect_hls +3..+4.

## Fix
Commit "MVAU_hls, LabelSelect_hls: per-frame HLS overhead in get_exp_cycles" (`5260cd099` on
`fix/002`): `MVAU_hls.get_exp_cycles()` = MVAU + 7, `LabelSelect_hls.get_exp_cycles()` =
LabelSelect + 4 (as dev's `RequantHLS` adds `hls_overhead = 10`); `tests/fpgadataflow/test_set_folding.py`
`min_cycles["U200"] = 16 + 7`. It does not belong to our PR branches (dev's estimate): the
orchestrator put it on a new standalone branch **`feature/dev/hls-exp-cycles-overhead`** (=
`upstream/dev` + this commit, `2169d7b6d`; `test_set_folding.py` passes there) and cherry-picked it
onto `test/{fifo,folding,dwc}` and `dev-finnexamples-features` (`tools/rebuild_branches.sh` merges it).

## Evidence
Offline re-run of the analytical FIFO sizing on 13 finished campaign builds, with/without the fix:
FIFO depths identical everywhere. Where an MVAU_hls is the slowest node, rtlsim = old estimate + 6..7:

| build | old est | new est | rtlsim | old ratio | new ratio |
|---|---|---|---|---|---|
| tfc-w2a2, tfc-w1a2 | 8 | 15 | 15 | 0.533 | 1.000 |
| cybersecurity-mlp | 1 | 8 | 6 | 0.167 | 1.33 |
| lfc-w1a1 | 512 | 519 | 518 | 0.988 | 1.002 |
| lfc-w1a2 | 2048 | 2055 | 2054 | 0.997 | 1.000 |
| kws | 3920 | 3927 | 3926 | 0.998 | 1.000 |
| gtsrb | 12288 | 12295 | 12294 | 1.000 | 1.000 |

cnv (all) and vgg10 unchanged (1.001-1.008). Folding runs: tfc-w2a2 fold-max 28 -> 35 (rtlsim 35),
tfc-w1a1 fold-max 14 -> 21 (rtlsim 20), integration optimizer run MVAU_hls_3 10 -> 17 (rtlsim 16).
Tests with the fix: `test_bnnpynq_fifo_sizing[...tfc-w2a2-heuristic_analytical]` and `[...tfc-w1a2-...]`
passed (ratio 1.000; 6.69 / 8.19 kbit). `test_set_folding.py`, `test_estimate_only_no_vivado.py` pass;
same src patch on the folding branch: test_set_folding 29 passed / 5 skipped / 1 xfail, on the DWC
branch 31/3/1; on fix/002 48 passed (analytical characterization mvau 12 + labelselect 8, chained TAV
2, 27 mvau internal_embedded rtlsim). Lint clean. Not run: `test_fifosizing_linear` (slow, <1 %
change), full `test_fpgadataflow_mvau.py`.

## Side experiment (diagnostic only, not applied)
`#pragma HLS DATAFLOW` on the generated MVAU top function brings the interval 15 -> 9 (trip + 1),
with a Vitis deadlock warning (ap_ctrl_none + auto-rewind); DATAFLOW + rewind without flp deadlocks
in rtlsim; rewind alone 15. Could nearly double tfc throughput; worth raising upstream (changes dev's
node implementation, deadlock risk). Variants: `build/finn_build/fx/fix002/hlsexp/`.

## Open points
* +7 is the measured upper bound except the INT8-threshold config (+8: 15 vs 16 = 0.94 at tiny trip
  counts); trip-1 designs up to 2 cycles conservative (cybersecurity 8 vs 6). +6 would be too low.
  Exact numbers would need the csynth interval after ipgen.
* Other HLS ops (VVAU_hls, Thresholding_hls, Pool_hls, ...) likely have the same overhead; not
  measured, none is a small-trip bottleneck in the current models.
* Scratch scripts: `/tmp/claude-1205/-home-lstasytis-finn/3f1e2c46-7f88-4375-a7c4-e46cd11c8455/scratchpad/i002/`
  (`multiframe.py`, `node_sim.py`, `sweep.py`, `fifo_cmp.py`); copied to `scripts/` next to this report.
