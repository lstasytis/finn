# Report 004: vgg10-radioml (ZCU104) folding target + padded DWC insertion — fixed
(Written by the orchestrator from the fix agent's hand-back.)

Branch `fix/004` (worktree `build/wt/fix004`), two commits, both `set_folding.py` +
`tests/fpgadataflow/test_set_folding.py`, both for `feature/dev/set-folding-optimizer` (no DWC /
InsertDWC change, `vpc.sv` untouched). Landed by the orchestrator on the folding PR branch (DWC
branch rebased) and cherry-picked onto `test/{folding,dwc}` and the integration branch.

| commit | what |
|---|---|
| d4a50a11c | SetFolding: find a DWC bottleneck in the repair loop (+ test) |
| 78520cf2e | SetFolding: no channel padding next to a reshape (+ test) |

## A: target missed (1280 vs 1028 cycles)
`SetFolding._bottleneck_nodes` keyed node cycles by name; DWCs inserted by `InsertDWC` during
scoring are unnamed, so they all shared key `''` (holding the last DWC's 24 cycles). The slowest
DWC (`ConvolutionInputGenerator_rtl_4 -> MVAU_rtl_2`, 128 -> 192 bit, 768 + 512 = 1280 cycles) was
never found; the repair aimed at `ConvolutionInputGenerator_rtl_0` (already maximally parallel) and
stopped. Not the cause: `Optimizer.dwc_cycles` and the DWC's `get_exp_cycles` agree; the
`compute_hls_dwc_cost` quirk only affects the LUT heuristic. Fix: compare cycles by position.

## B: `Pool_hls_6 -> MVAU_rtl_7: DWC can only pad/crop within a row's last word`
With `folding_maximum_padding=4`, polish padded `MVAU_rtl_7` (dense layer after the flatten) MW
256 -> 259; its producer sends 8 rows of 32, the MVAU reads 1 row of 259; a padding DWC pads/crops
row ends, so both sides need the same row count. Fixing it in the DWC would be wrong in general
(per-row cropping of a padded producer). Fix: `_max_padding_for_node` gives no padding budget to a
node bordering a node with a different row count (`_borders_reshape`; for SWG also the node it
feeds); only runs when padding is enabled.

## Evidence
* `test_vgg10radioml_folding[ZCU104-target]` passed: estimate 243190.66 fps (1028 cycles = target),
  rtlsim 243427 fps (1.001x), verified (`fx/fix004/fold-target_vgg10-radioml_ZCU104_yc7mj2w6`).
* `test_vgg10radioml_dwc[ZCU104]` skips with `folding_seed=1` (padding not worth it); before it
  crashed in InsertDWC. With seed 4 (seeds 4 and 8 pad): 2 padding DWCs (32 -> 36 channels around
  SWG_11/Pool_hls_5), target met, rtlsim 1.000x, verified (`fx/fix004/dwc-s4_vgg10-radioml_ZCU104_b_4lv5v3`).
* test_set_folding 36 passed/1 xfail, test_fpgadataflow_dwc 88 passed; 2 new tests fail on the old file.
* Gate: fresh baselines (`fx/refactor/dumps/fix004base`, `fix004basex`; the old `base1` is stale
  since the estimate refresh and the int cast) vs fix (`fix004`, `fix004x`), 108 + 52 runs: only
  `vgg10__padded` and `__padio` differ (AssertionError -> ok at 1028 cycles).

## Open points
* DWC estimate pessimistic for non-integer ratios / padding (in + out words; rtlsim 1088 vs 1280).
  Would be a DWC-branch change (`get_exp_cycles`, `Optimizer.dwc_cycles`).
* `test_vgg10radioml_dwc` skips with seed 1; `folding_seed=4` would always exercise padding (test decision).
* Future gates: use `fix004base`/`fix004basex` or rerun the parent file.
