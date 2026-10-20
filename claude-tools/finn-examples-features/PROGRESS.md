# PROGRESS — finn-examples per-model feature tests (session of 2026-10-19)

Executed `claude-tools/finn-examples-features/PLAN_per_model_tests.md` in one day; the machine
reboots at ~20:00, the campaign was stopped at 19:10. **Results table:
`claude-tools/finn-examples-features/STATUS.md`**; raw results:
`claude-tools/finn-examples-features/results/campaign_20261019/`; end-of-day handoff:
`claude-tools/finn-examples-features/HANDOFF.md` section 0. (A copy of this file is committed as
`claude-tools/finn-examples-features/PROGRESS.md`.)

## How to resume after the reboot

1. Container: on the host, `./restart.sh` (see its header; if `./restart.sh save` was run before
   the reboot it starts the snapshot, otherwise restore `$HOME` from
   `build/home_backup/claude_home.tar` as it prints). Then `cd /home/lstasytis/finn && claude --continue`.
2. Read this file, `claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, branches,
   rules) and `STATUS.md`.
3. Worktrees survive in `build/wt/*` (repo volume). Re-run `tools/wtsetup.sh <wt>` if a symlink is
   missing. The session scratchpad (diagnostic data of the fix agents, e.g. issue 007's `ctx_hw.npz`)
   was copied to `build/home_backup/scratchpad/`; all of `/tmp/claude-1205` (incl. older sessions' scratch dirs) to `build/home_backup/tmp_claude-1205/` (removed from the container to make `restart.sh save` fit).
4. Restart the campaign for what is left (aborted jobs are rerun; failed ones only with
   `--rerun-failed`; requeue single jobs with `tools/requeue.sh <job ids>`):
   ```
   cd /home/lstasytis/finn/build/finn_build/fx/logs/campaign
   (setsid nohup python3 /home/lstasytis/finn/claude-tools/finn-examples-features/tools/run_campaign.py \
       --max-threads 160 --deadline <HH:MM> >> runner.log 2>&1 < /dev/null &)   # writes runner.pid itself
   ```
   The test branches already contain every fix (cherry-picked). Jobs to rerun first: all folding /
   DWC jobs of resnet18, resnet50, yolov8, bert (fix 003 landed at 18:45, after they had failed),
   the aborted large-model FIFO jobs, espcn (issue 007).

## What was done

* **Phase 0** env (`tools/env24.sh`, `tools/wtenv.sh`, `tools/wtsetup.sh`), models/goldens
  unzipped, BNN-PYNQ models re-exported (the 2020 ONNX files broke the preprocessing), xsi built,
  lint env `.lintenv`, `GENERAL.md`.
* **Phase 1** `dev-finnexamples-features` = DWC + FIFO + #1725 + HLS-overhead fix + tests;
  `test/{fifo,folding,dwc}` (`tools/rebuild_branches.sh`). Since 12:00 fixes were cherry-picked
  onto the test and integration branches.
* **Phase 2** per-model feature tests in all 11 model files (markers `fifo_sizing`, `folding`,
  `dwc`; helpers in `benchmark_helpers.py`; README section; old harness deleted), ported to the PR
  branches (own tests only, dev's models only; folding/DWC there use dev's `characterize` FIFO
  strategy). Smoke test on tfc-w1a1 found the issues fixed below.
* **Phase 3** folding refactor: `set_folding.py` 2899 -> 1949 lines (target ~1500 not reached:
  dev's naive folder, docstrings, features), byte-identical dumps on 108 + 50 runs
  (`agents/folding_refactor/REPORT.md`).
* **Phase 4** campaign: 110 jobs (`tools/run_campaign.py`), see STATUS.md.
* **Phase 5** seven issues, each with ISSUE.md + REPORT.md in `claude-tools/finn-examples-features/issues/`:

| # | issue | result | branch |
|---|---|---|---|
| 001 | bert FIFO sizing: dynamic MVAU rtlsim characterization timed out | fixed | FIFO |
| 002 | tfc/MLP rtlsim 0.53x estimate: MVAU_hls/LabelSelect_hls per-frame HLS overhead | fixed (+7/+4 cycles) | new `feature/dev/hls-exp-cycles-overhead` |
| 003 | optimizer: SWG pairing in branching graphs (resnet18/50, yolov8), bert dynamic MVAU | fixed 18:45, big models not rerun | folding + 2 InsertDWC fixes on DWC |
| 004 | vgg10: unnamed DWC bottleneck missed, padding next to a reshape | fixed | folding |
| 005 | maximize on ZU3: phantom URAM of embedded weights + search fixes | fixed | folding (MVAU line is also a dev bug) |
| 006 | balanced foldings 2-6 % below estimate: FIFO sizing | fixed (pacer-bound chains, refill floor) | FIFO |
| 007 | espcn stitched-IP rtlsim: whole pixels zeroed under backpressure | **open**, narrowed down | (probably dev nodes) |

Builder changes on the folding PR branch needed by the tests: maximize mode reachable
(`folding_style="optimizer"` without target/JSON), `folding_seed`, platforms KV260_SOM/VCK190/V80.

## Results (STATUS.md has every cell)

* FIFO `heuristic_analytical`: passes (correct, rtlsim/estimate 0.99-1.33) on every model whose job
  finished: all bnn-pynq (10), gtsrb, kws, cybersecurity-mlp, vgg10, mobilenet_v1 (U55C), bert
  (rtlsim 4x faster than the estimate: tiled-MVAU estimate pessimistic). resnet18_8w8a n/a (URAM
  runtime weights in rtlsim). espcn fails (#007). resnet18_4w4a, yolov8 x2, resnet50: aborted at the
  deadline (stitched-IP rtlsim of these takes > 7 h).
* FIFO `heuristic_rtlsim`: passes on all small models, mobilenet_v1 and bert.
* Folding target: passes on all small models except tfc-w1a1 (target infeasible: committed folding
  near-fully unrolled beyond the optimizer's ZU3 budget, LabelSelect kept at PE=1 — user decision),
  mobilenet passes; large models need a rerun after #003.
* Folding maximize: passes on all small models (cnv up to 21x the committed throughput on U55C);
  mobilenet maximize (15.7x committed) rtlsim 0.85-0.88x estimate (~80 nodes at exactly the target
  period: optimizer margin question).
* DWC: padding pays off at the reference target only on vgg10 (2 padding DWCs, passes) and
  mobilenet (16, rtlsim 0.976x, FIFO edge not found); all other models skip (seeds 1-8 tried).

## Open items / user decisions
1. tfc-w1a1 folding target infeasible by design (cap the target or accept).
2. Issue 007 (espcn), resume steps in its REPORT.md.
3. Upstream findings (not fixed): BIPOLAR weights widened to INT2 by `minimize_weight_bit_width`;
   FINN's LUT model for binary HLS MVAUs several times pessimistic; tiled-MVAU `get_exp_cycles` 4x
   pessimistic (bert); VVAU_hls per-frame overhead (+8..10); `OuterShuffle_hls` reaches II=3 (3x its
   estimate, espcn); `#pragma HLS DATAFLOW` on MVAU_hls would nearly double small-MLP throughput.
4. FIFO rules of #006 are conservative (+25 % storage on some foldings without gain).
5. Maximize / FIFO-aware folding on yolov8/resnet50/bert takes > 4 h.
6. resnet18_8w8a cannot fit U55C without URAM weights (`folding_allow_uram_weights`).

## Push commands (for the user; nothing was pushed)

```
git push origin feature/dev/analytical-fifo-sizing feature/dev/set-folding-optimizer
git push origin feature/dev/hls-exp-cycles-overhead
git push --force-with-lease origin feature/dev/generalized-datawidthconverter dev-finnexamples-features
git push origin claude-tools
```
(The DWC branch was rebased onto the new folding tip and the integration branch was rebuilt, hence
the force push. `test/*` and `fix/*` branches are local working branches, no need to push.)

## Session 2 (2026-10-20, after the reboot)

* 10:48 campaign restarted (`--deadline none --max-threads 160 --only '^(?!(fifo|fold-target)__espcn)'`;
  espcn FIFO/fold-target excluded while #007 is open). Requeued the 18 folding/DWC jobs of resnet18 x2,
  yolov8 x2, bert, resnet50 that failed before fix #003 (old logs in `logs/campaign/superseded/*.1020.*`);
  aborted jobs rerun automatically. Runner now also gates on summed memory reservations and on
  >= 60 GB free disk (volume at 96 %).
* Issue 007 (espcn) handed to a fix agent again (session 2 section in `ISSUE.md`).
* Not requeued (no fix yet): mobilenet fold-max (0.85x) and DWC (0.976x), see #006 notes.
* 11:20 user pulled upstream: `upstream/dev` +48 commits (#1729: SetFolding folds RTL elementwise,
  Requant, HWSoftmax, generic Shuffle), `upstream/update/finn-examples` (#1725) + finn+ transformers
  (`tests/benchmark/transformer/`, models + verification npys copied from the repo root into
  `tests/benchmark/{models,verification_io}`), `upstream/fix/exp_cycles` (tiled MVAU + FINNLoop).
* Fix 008 (folding speed, user: transformers must fold as fast as regular models, branches split
  like chains): `issues/008-folding-speed/REPORT.md`; on `feature/dev/set-folding-optimizer`
  (`4b09bdfd3`, Shuffle in the optimizer `81b625ae3`).
* All PR branches merged with the new `upstream/dev`; integration branch = DWC + FIFO + HLS + new
  #1725 + transformer feature tests (`4ed8cb505`); new test branches `test2/{fifo,folding,dwc}` in
  worktrees `build/wt/{fifo2,folding2,dwc2}` (`SUFFIX=2 rebuild_branches.sh`).
* Campaign variant 2 (`run_campaign.py --variant 2`, job ids `v2__*`): transformers first, then the
  big-model folding/DWC jobs, then the rest; bert held for issue 009 (tiled MVAU exp cycles, fix
  agent running), espcn FIFO/fold-target held for issue 007 (fix agent running). Old-base jobs that
  were running were adopted and finish on the old base.
* Fix 009 (tiled MVAU exp cycles) landed: new `feature/dev/mvau-exp-cycles` (upstream c15fb8a9a + group
  stalls, for dev); FIFO branch fast-forwarded to `fix/009-fifo` (tree model uses the shared stall);
  merged into integration + test2/*; bert estimate = rtlsim (98304, was 4x). bert v2 jobs released.
* Fix: optimizer offered URAM for internal_embedded constants off Versal (transformer_vision ZCU104),
  `57594bf0c` on the folding branch, propagated.
* Open: transformer_benchmark rtlsim 0.965x estimate (identical interval for two foldings: MVAU_rtl
  SIMD=PE=1 per-vector overhead?), sent to the 009 agent; old-base bert fold-target deadlocked in
  stitched-IP rtlsim (rerun on v2 with the new estimates); issue 007 tapped sims running.
* User (afternoon): do the 3 updates (issue 012: MVU queue + Split/Concat pragma as separate dev PR branches with
  PR_*.md; 013: latency in the FIFO sizer, over-estimates ok within 1.5x of optimum; 014: DWC padding should
  tighten over-throughput nodes, new acceptance rule <=10 % below target if resource saving % > throughput loss %).
  Agents on Opus (user: no Fable), <=10-line hand-backs.
* Feature tests made self-contained per model file (helpers replicated, benchmark_helpers.py = upstream) on the
  integration, PR and test2 branches (`tools/inline_feature_helpers.py`). Small-folding tier (`_small.json`,
  validated against docs/finn/reference/folding-constraints.rst by `tools/check_small_foldings.py`) runs first,
  full jobs only after their small sibling passed (`run_campaign.py --sizes small,full`).
* Issue 010 fixed (Split/Concat 5 cycles/word estimate + optimizer folding, FIFO phase lead), transformer_benchmark
  FIFO/fold-target pass at 1.00x; maximize still 0.73x (needs issue 013).
* FIFO-aware folding flag (`enable_folding_fifo_heuristic`), one check on gtsrb small: correct, 0.991x, FIFOs -25 %,
  folding 125 s vs 6 s (~20x) -> stays off in the main flow (user: optional).
* Open: issue 011 (radioml/vision hangs + wrong outputs), 012-014 agents running.
