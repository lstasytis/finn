# Report 023: land 018, LabelSelect PE under the optimizer, requeue the campaign

(Another agent created `issues/023-islands-benchmark/` at the same time; the number collides, the
folder name was given by the user.)

## 1. Issue 018 landed (rule "Decoupled chain")

* `feature/dev/analytical-fifo-sizing` fast-forwarded to `fix/018` (`e3723ec90`, `6f734bc6e`;
  worktree `build/wt/pr-fifo`).
* Merged (as the earlier FIFO fixes) into `dev-finnexamples-features` (`30ce8fa34`), `test2/fifo`
  (`9f8449048`), `test2/folding` (`45ad7d078`), `test2/dwc` (`ee2c81c2e`): all three test2 branches
  carry the FIFO sizer. No conflicts; DWC invariant holds on integ and dwc2 (one
  `get_exp_cycles`, tree-model guard present).
* `tests/fpgadataflow/test_fifosizing_chained_tav.py`: 8 passed on the FIFO branch and on integ.
* rtlsim spot checks (`fx/fix018/res7`, finished after hand-back): espcn interval 783369 cycles
  base and 018 (FIFO bits 120.63 M -> 120.62 M), resnet18 100978 both (2.292 M -> 2.279 M bits);
  both complete, no timeout: same throughput with slightly smaller FIFOs.

## 2. LabelSelect PE folds under the optimizer

* Rule: `("LabelSelect_hls", "PE")` in `Optimizer.DO_NOT_FOLD`, added because LabelSelect's fmax
  drops with PE (OOC P&R xczu7ev: 10 labels 610 / 390 / 230 MHz at PE 1 / 2 / 5; HANDOFF section
  5). That is its only reason: the HLS `LabelSelect_Batch` itself only needs PE | Labels (block loop
  `NumClasses / PECount`), and the stream width PE x bits is covered by the generic HLS width limit.
* Removing the entry alone was not enough: `max_folding_factor` bounds PE by the output's last dim,
  which for LabelSelect is K (1), not Labels. New `Optimizer.PE_MAX_OVERRIDE` maps
  `LabelSelect_hls` to `Labels`; LabelSelect is not paddable, so the offered values are the divisors
  of Labels. The fmax measurement stays as a comment next to the override.
* Behaviour (MVAU 16x10 -> LabelSelect 10 labels, U250): LabelSelect stays at PE=1 when the target
  allows it (targets 16 and 10 cycles), PE=2 / 5 at targets 5 / 3, PE=10 in maximize mode.
* Test `test_set_folding_labelselect_pe[maximize|target]`: offered PE values {1,2,5,10}; the result
  divides Labels, is >= 5 and meets the target / beats LabelSelect's PE=1 cycles. The target is
  computed from the node models so it also holds with the integration branch's per-frame HLS
  overheads (the first version, `80bd64832`, failed there with a fixed 2-cycle target; fixed in
  `54fa8d2b5`). Both cases fail without the change.
* Commits on `feature/dev/set-folding-optimizer` (on top of 021's `00951c1e7`, `f3e0dfade`):
  `80bd64832`, `54fa8d2b5`. Landed like the earlier set_folding fixes: merged into
  `feature/dev/generalized-datawidthconverter` (`8252bb6d2`, `6ea413550`), that into
  `dev-finnexamples-features` (`0a67879e7`, `86e1897c1`), cherry-picked (-x) to `test2/folding`
  (`f4f3a9409`, `6326db5f9`) and `test2/dwc` (`0fec95f6c`, `f6ba08c1b`).
* tfc-w1a1 AUP-ZU3 maximize, estimate only on test2/folding (`fx/fix023/chk-max_*`): 12.5 M fps =
  **1.125x** the committed folding (was 0.60x); LabelSelect PE=5 (6 cycles with the HLS overhead),
  the four MVAUs and the input DWC/Thresholding at 8 cycles. Campaign rerun
  (`fx/folding2/fold-max_bnn-pynq_tfc-w1a1_AUP-ZU3_8GB_pkcsty7c`): output verified, but rtlsim
  11.11 M fps = 0.889x its estimate (9 cycles/frame instead of 8), so the test FAILS its 0.98
  rtlsim/est. check; rtlsim is exactly the committed folding's estimate (1.00x ref). Some node
  takes one cycle more per frame than modelled (LabelSelect at PE=5 or an MVAU at 8): open.
* `tests/fpgadataflow/test_set_folding.py`: folding branch 52 passed / 5 skipped / 1 xfailed (the 2
  new ones passed separately); integ 57 / 1 xfailed; test2/dwc 57 / 1 xfailed; test2/folding 55 /
  2 skipped / 1 xfailed. black/isort/flake8 clean.

## 3. Campaign requeue

### Reference estimates (est_* dirs)
* The tests cache the committed-folding estimate as `$FINN_BUILD_DIR/est_<model>_[small_]...`
  (`cached_build` -> `find_cached_build`: the first dir by name with the prefix, rebuilt if it lacks
  `report/estimate_network_performance.json`). Every est dir in `fx/folding2` and `fx/dwc2`
  predates issue 021's estimate model in those worktrees (2026-10-22 04:29:13 UTC, last estimate
  commit 55c9bb8a2): 110 dirs. fifo2's est dirs are left (FIFO tests use only their checkpoint and
  fps).
* Moved to `fx/<wt>/superseded_021/` as soon as no running job of that worktree uses the prefix
  (86 at once, the rest when their jobs end); the two resnet50 small ones from issue 020
  (`folding2/est_resnet50_small_V80_lclp_otw`, `dwc2/est_resnet50_small_V80_3szgb2zd`) go to
  `superseded_020/` once fold-target / dwc resnet50 small end.
* Groups shared by running and pending jobs (folding2 full: resnet18 8w8a, yolov8 4w4a / 8w8a,
  resnet50, espcn) got a fresh estimate now via `tools/regen_reference_estimate.py`, named
  `<prefix>00000021`, which sorts before every random suffix: new jobs use it while running jobs
  keep the old one.
* Incident: the first watcher run also treated dirs without a report as stale and moved 5 est dirs
  that jobs launched by the runner restart were building at that moment (bert small/full,
  transformer_benchmark small, in folding2/dwc2). Only early-step files had moved; they were merged
  back (no overlapping files), all 5 dirs are complete. `v2__fold-target__bert__VCK190__small`
  failed on it (missing dataflow_parent.onnx) and was requeued; the other jobs had their checkpoint
  in place. Fixed: a dir without a report is stale only if it is older than the cutoff.

### Jobs
* Watcher `scripts/requeue_watch.py` (PID 2966019 after the restart; log
  `fx/logs/campaign/requeue_023_watch.log`, remaining list `requeue_023_watch.json`).
* Requeued at 09:41 (103 jobs): all finished `v2__fold-max__*`, `v2__fold-target__*`, `v2__dwc__*`
  (except transformer_radioml/vision/language) and the finished `v2__fifo__*` /
  `v2__fifo-rtlsim__*` of mobilenet_v1, vgg10-radioml, resnet18, resnet50, espcn, bert.
* Requeued when they end (21 running jobs, not killed): dwc bert, espcn small, mobilenet_v1,
  resnet18 4w4a/8w8a small; fold-max espcn, resnet18 8w8a, resnet50, yolov8 x2; fold-target resnet50
  small, yolov8 x2 small; fifo espcn small, resnet50 small; fifo-rtlsim espcn small, resnet18 4w4a,
  resnet50 small. The three dwc jobs (yolov8 x2 small, resnet50 small) that issue 022's watcher
  (PID 19389) requeues are left to it; mine only moves their est dirs first (it polls every 15 s,
  022's waits >= 60 s).
* The `--only` exclusion is kept: requeue.sh restarts the runner with the live runner's arguments.
* Not requeued (not in the request): `v2__fold-fifo__*`, which also use the optimizer and the old
  estimates.

### Tool changes (claude-tools)
* `tools/requeue.sh`: `flock` on `requeue.lock` (two watchers call it), the lock fd closed for the
  restarted runner; the live arguments are written to `runner.args`.
* `tools/regen_reference_estimate.py`: fresh reference estimate under a name found first.
* `tools/ref_resources.py`: res./ref. reference = each FIFO test build (committed folding + DWCs)
  re-estimated with the folding2 code, stored as `fx/ref_resources/<fifo job>.json`. The FIFO
  builds ran on test2/fifo, which does not contain `feature/dev/resource-estimates`, so their own
  `estimate_layer_resources.json` is the old model (gtsrb 76.2 k LUT / 314 BRAM_18K vs 51.9 k / 178
  re-estimated; tfc-w1a1 230 k vs 38.2 k LUT). 40 of 52 FIFO jobs have a build; the rest are
  running, pending, blocked or skip by design (resnet18 8w8a).
* `tools/make_summary_tex.py`: runs `ref_resources.py` first (`--no-ref-update` skips it). A
  folding/DWC build started after the cutoff (first timestamp of its build_dataflow.log) is compared
  with the re-estimated reference, else with the estimate-only build (dagger, no DWCs) if that one is
  new-model, else the cell stays empty; a build started before the cutoff is compared with the FIFO
  build's own (old-model) report. The small-size fallback lookup used a wrong prefix for
  bnn-pynq/resnet18/yolov8/transformer/mobilenet; `status_table.est_prefix` builds it as the tests do.
* `tools/status_table.py` has no res./ref. column (its pad/no-pad ratio compares two estimates of
  the same run); only `est_prefix` was added.
* GENERAL.md: note on est dirs and the reference resources.

## Open points
* tfc-w1a1 maximize: 9 vs 8 estimated cycles/frame in rtlsim (see section 2); find the node.
* `v2__fold-fifo__*` jobs: same staleness as fold-max/target; requeue if wanted.
* test2/fifo lacks `feature/dev/resource-estimates` (rebuild_branches.sh includes it): merge it at
  the next rebuild; new FIFO builds then report the new model directly.
* LabelSelect fmax: 230 MHz at PE=5 for 10 labels (above these builds' clocks); with 1000 labels a
  high PE would limit the clock.
