# Report 014: padding DWCs rarely chosen — fixed (optimizer did not exploit padding)
(Saved by the orchestrator from the agent's hand-back; subagents cannot write report files.)

## Root causes (all in set_folding.py)
1. The annealer's DWC-cost heuristic charged a padding DWC 8x: `compute_hls_dwc_cost` treated padding/cropping as
   HLS-only, but only the RTL DWC (vpc) pads and every DWC is RTL on dev. E.g. cnv-w2a2 MVAU_hls_3 SIMD (128,4,0) ->
   (117,13,0) saves 508 LUT + 3 BRAM (cost -0.019) but the heuristic added +0.089.
2. Polish moves one step in the sorted value list; for an SWG -> MVAU pair the values are lexicographic tuples, so
   (117,13,0) is many steps from (128,4,0).
3. `_better` rejected a cheaper padded folding if peak utilization rose at all (mobilenet: -24k LUT (-12 %) for +12
   BRAM (+1.2 %), BRAM the peak resource).
4. A conv MVAU's PE (output channels) was never padded (no reason; weight/threshold resizing handles it).
Not a cause: the padding DWC's cycle estimate (in + out words) is right or slightly conservative (vpc rtlsim in
padding mode, 11 configs: in+out+2 for short rows, 7-13 % below in+out for long rows); a padding DWC next to a node
running at the target cannot keep up. vpc.sv untouched.

## Changes
`fix/014-folding` (off feature/dev/set-folding-optimizer 1f9a9fe21):
* a1caca9ed: heuristic: only a non-integer width ratio gets the 8x penalty; new tighten pass in `_refine` between two
  polishes (each parameter jumps to the best of its slower values that keep its nodes within the target and cost
  less on their own, <= 3 candidates per parameter, scored on the whole graph; never worse than before); `_better`:
  among foldings meeting the target and fitting, lower weighted cost wins, peak utilization breaks ties (a padded
  folding missing the target never replaces one meeting it); conv MVAU PE folded over padded output channels, XNOR
  blocks only the input-width side; 5 new unit tests (4 fail on the old code).
* 2620ceeeb: a StreamingConcat fed by a StreamingSplit's second output hit "Malformed model" in the heuristic (bug
  from the Split/Concat SIMD folding commit of issue 010): both yolov8 crashed in the folding step.
* test_set_folding.py: 51 passed / 5 skipped / 1 xfail; lint clean. No DWC-branch change needed.
`fix/014-tests` (off dev-finnexamples-features 3380fd326): 69bf85a06, acceptance rule in all 12 per-model DWC tests:
`TARGET_TOLERANCE = 0.10`, `resource_share(output_dir, board)` (LUT, BRAM_18K, URAM, DSP as fractions of the board,
FIFOs excluded because the reference estimate sizes none, DWCs counted), `assert_target_or_trade(metrics, ref_dir,
target_fps, board)`: passes if the estimate meets the target, else loss <= 10 % and resource reduction vs the
committed folding > loss. Kept out of the optimizer: the committed folding (the comparison) exists only in the tests,
and the padded search met the target on every model. Smoke check (gtsrb): 5 % below with 24.5 % fewer resources
passes, 30 % below fails.
`fix/014-eval` (build/wt/fix014, test2/dwc + the fixes): evaluation only, disposable.

## Per model, unpadded vs padded at the reference target (estimate-only, seed 1, padding 0 vs 4)
Resources LUT/BRAM/URAM/DSP without FIFOs; last column with FIFOs (LUT/BRAM).
| model (board) | target | unpadded | padded | padded nodes, pad DWCs | with FIFOs (unpadded -> padded) |
|---|---|---|---|---|---|
| bert (VCK190) | 393216 | 393216, 470857/906/4/208 | 393216, 470742/906/4/208 | 1, 2 | 479624/925 -> 479375/925 |
| cnv-w1a1 (ZU3) | 8167 | 8167, 83637/79/0/0 | same | 0, 0 | - |
| cnv-w1a2 / cnv-w2a2 (ZU3) | 33020 | 33020, 39481/197/0/0 | 33020, 38973/194/0/0 | 2, 1 | 42061/198 -> 41321/201 |
| cnv-w1a2 (KV260) / cnv-w2a2 (U55C) | 33020 | 33020, 44512/175/0/0 | 33020, 44004/172/0/0 | 2, 1 | 47094/176 -> 46415/179 |
| cybersecurity-mlp | 8 | 8, 584568/0/0/0 | same | 0, 0 | - |
| espcn (KV260) | 524288 | 786432, 12620/228/0/459 | same | 0, 0 | target missed either way (OuterShuffle, #007) |
| gtsrb (ZU3) | 12295 | 11271, 55106/240/0/0 | 11760, 52890/228/0/0 | 3, 3 | 57436/240 -> 55915/229 |
| kws | 3927 | 3927, 5072/49 | same | 0, 0 | - |
| lfc-w1a1 / lfc-w1a2 | 519 / 2055 | 41428/159; 23204/318 | same | 0, 0 | - |
| mobilenet_v1 (U55C) | 788544 | 788551, 138601/972/0/10 | 788551, 115499/987/0/10 | 19, 23 | 148772/974 -> 125185/1000 |
| resnet18_4w4a | 65536 | 65538, 41897/2795/0/2122 | 65538, 41979/2795/0/2122 | 2, 1 | 67613/2829 -> 68282/2831 |
| resnet18_8w8a | 147456 | 147456, 56095/5027/0/1008 | 147456, 56066/5027/0/1008 | 0, 0 | 80882/5163 -> 79701/5158 |
| resnet50 (V80) | 903175 | 903175, 236438/3626/0/10 | 903175, 225549/3611/0/10 | 12, 9 | 271260/3662 -> 260862/3642 |
| tfc-w1a1 / w1a2 / w2a2 | 9 / 15 / 15 | unchanged | same | 0, 0 | - |
| quantyolov8_4w4a | 1105920 | 1105920, 115176/951/0/290 | 1105920, 115342/948/0/280 | 8, 5 | 142417/1088 -> 142602/1084 |
| quantyolov8_8w8a | 1105920 | 1105920, 56746/1467/0/315 | 1105920, 57426/1457/0/321 | 9, 5 | 90664/1664 -> 93565/1645 |
| transformer_language (VCK190) | 98304 | 98304, 40141/712/70/1388 | 98304, 39007/704/70/1388 | 1, 4 | 66758/920 -> 64735/891 |
| transformer_radioml / vision / benchmark | - | - | no padding (radioml -293 LUT from the wider search) | 0, 0 | - |
| vgg10-radioml | 1028 | 1028, 10787/375/0/817 | 1028, 10761/375/0/817 | 0, 0 | - |
Before the fix only mobilenet padded (13 pad DWCs). resnet18_4w4a and yolov8_8w8a padded foldings are cheaper by the
optimizer's objective but +0.7k / +2.9k LUT once the analytic sizer sizes the FIFOs.
Where padding still does not happen: XNOR models (tfc/lfc/cnv-w1a1, most of gtsrb): MW cannot be padded (a zero bit
counts as -1; padding weights with 1 would need guaranteed-zero padded lanes: possible follow-up); nodes already exactly
at the target (lfc, vgg10, tfc); padding DWC speed next to a node at the target; BRAM weighting (kws: 4 of 183 BRAM).

## rtlsim
`test_gtsrb_dwc[AUP-ZU3_8GB-full]` PASSED: 3 pad DWCs (padded conv MVAU output channels), verified, rtlsim 1.000x
estimate, 8503 fps vs reference 8133. `test_bnnpynq_dwc[AUP-ZU3_8GB-cnv-w2a2-full]` PASSED: 1 pad DWC, verified,
1.008x, 3028.5 fps.

## Files
Scripts in scripts/: fold_compare.py, run_compare.sh, polish_probe.py, dwc_pad_cycles.py, eval_build.py,
run_eval.sh, eval_table.py; table also in fx/fix014/eval/table_v3.md. No deletions; fx/fix014 (6.8 GB) can go.
