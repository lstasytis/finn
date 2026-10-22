# 022: DWC tests compare the folding optimizer without and with padding

## What changed

`test_<m>_dwc` in every `tests/benchmark/<model>/test_build_<model>.py` (helpers replicated per
file; applied by `tools/add_dwc_pad_compare.py <worktree>`, then isort + black):

1. **Target mode** (reference target = committed folding's estimated fps): for each seed in
   `DWC_SEEDS` (1-8) two estimate-only optimizer builds with the same seed,
   `folding_maximum_padding=0` (builds `dwc-est-nopad*`) and `=FOLDING_MAX_PADDING` (4,
   `dwc-est*`). Resource ratio padded/unpadded = mean of the LUT/BRAM_18K/URAM/DSP ratios over
   the resources the unpadded folding uses (FIFOs excluded, DWCs included: both estimate builds
   run analytic FIFO sizing, which inserts the DWCs) = the res./ref. metric of
   `make_summary_tex.py`. The seed with the smallest ratio is chosen (a padded folding using a
   resource kind the unpadded one does not is ranked last; on a tie the one with padding DWCs).
2. **Maximize mode** (full size only): estimate-only, no target, seed 1, padding off vs on
   (`dwc-max-nopad*`, `dwc-max*`): fps ratio padded/unpadded (warning if < 1).
3. **Hardware**: the chosen padded folding is built (stitched IP + rtlsim) as before, with
   `assert_target_or_trade`. No skip any more: when padding saves nothing the verdict is
   recorded and the test passes (the hardware build still runs, it then checks the padded
   optimizer flow without padding DWCs). `assert_padding_not_worse`: warning if the padded
   folding needs more resources than the unpadded one, failure above 5 %
   (`PADDING_WORSE_TOLERANCE`).
4. **Metrics**: `report/feature_metrics.json` of the `dwc*` build holds `pad_compare`:
   `seed`, `target_fps`, `res_ratio`, `res_extra`, `verdict` (`gain`, `no gain`,
   `no gain (no padding DWC)`, `worse`), `nopad`/`pad` (`est_fps`, `pad_dwcs`, `resources`,
   `output_dir`), `seeds` (ratio and pad DWCs per seed) and, full size, `maximize`
   (`fps_ratio`, `res_ratio`, `res_extra`, `nopad`/`pad`). `run_feature_build` got
   `extra_metrics=None` (stored before its asserts, so also on a failing hardware build).

Cost: 16 estimate-only builds per test (was up to 8, stopping at the first padded one) plus 2
maximize estimates at full size; every test now builds hardware (most models used to skip), so
e.g. espcn/resnet18 full dwc jobs now take as long as their fold-target jobs.

## Commits

* `dev-finnexamples-features` 099e27415 (where the per-model tests live), cherry-picked (-x) to
  `test2/fifo` 43d3475da, `test2/folding` f4a5bdbbb, `test2/dwc` cf0d4687a, on top of the other
  agents' small-folding commits (nothing rewritten). Not ported to the `feature/dev/*` branches
  (the earlier DWC trade-off change was not either).
* claude-tools: `tools/add_dwc_pad_compare.py`; `tools/status_table.py` (STATUS.md columns
  "DWC pad/no-pad res. (target)" with verdict and seed, "DWC pad/no-pad fps (maximize)" with the
  padded maximize folding's pad DWCs, "DWC #pad DWCs"; small / full); `tools/make_summary_tex.py`
  (DWC table: small = #pad DWCs, res. pad/no-pad, rtlsim/ref., res./ref.; full adds fps
  pad/no-pad (max.)); `tools/requeue.sh` restarts the runner with the live runner's exact
  arguments (`/proc/<pid>/cmdline`) instead of `runner.args`, which lagged behind the running
  runner (another agent's cnv-w1a2/w2a2 exclusion would have been lost); GENERAL.md;
  regenerated STATUS.md and summary/summary.tex (no LaTeX on the machine: not compiled).

## Campaign

* 2026-10-21 23:44: requeued all 32 finished `v2__dwc__*` jobs except
  transformer_radioml/vision/language; the runner restarted with its live arguments (`--only`
  filter excluding those transformers, and another agent's temporary cnv-w1a2/w2a2
  fold-target/dwc exclusion, kept).
* Running dwc jobs left alone: `v2__dwc__yolov8__U55C-quantyolov8_{4w4a,8w8a}__small` (pids
  1955950, 1956768), `v2__dwc__resnet50__V80__small` (144332). A detached watcher
  (`scripts/requeue_when_done.sh`, log `build/finn_build/fx/logs/campaign/requeue_dwc_watch.log`)
  requeues each 60 s after its process exits. The pending full yolov8/resnet50 dwc jobs start
  with the new code anyway.

## Early result: vgg10-radioml small (ZCU104), local run in build/wt/dwc2

PASSED in 30 min (`dwc-small_vgg10-radioml_ZCU104_2xlro1v2`). Target 30458 fps.

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| res. pad/no-pad | 1.000 | 0.995 | 0.992 | **0.984** | 0.988 | 1.000 | 1.000 | 1.000 |
| pad DWCs | 0 | 0 | 4 | 3 | 5 | 0 | 0 | 0 |

Seed 4 (verdict gain): unpadded LUT 6492, BRAM_18K 66, DSP 112; padded LUT 7043 (+8.5 %),
BRAM_18K 62 (-6 %), DSP 104 (-7 %), both at 30458 fps; hardware: 3 pad DWCs, rtlsim 1.000x
its estimate, estimate 1.00x the committed folding. Note: the mean-ratio metric counts the
LUT increase against the BRAM/DSP savings equally; seed 2 (0.995 without padding DWCs) shows
the annealing noise level (~0.5 %).
