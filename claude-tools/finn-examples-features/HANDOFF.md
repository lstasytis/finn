# finn-examples feature flows: FIFO sizing, folding, DWC padding (handoff)

## 0. State at the end of 2026-10-19 (per-model feature tests, read this first)

The plan `PLAN_per_model_tests.md` was executed in one day (machine reboot at 20:00). Live results:
**`STATUS.md`** (models x features table), exported raw results: `results/campaign_20261019/`
(per job: junit XML, pytest log tail, feature metrics and reports of the build), issue manuals and
fix reports: `issues/NNN-*/{ISSUE,REPORT}.md`, repo primer for agents: `GENERAL.md`, day log and
resume instructions: `PROGRESS.md` (copy of the repo-root file).

* **Tests:** the generic feature-flow harness is gone; each `tests/benchmark/<model>/test_build_<model>.py`
  has `test_<m>_fifo_sizing[heuristic_analytical|heuristic_rtlsim]`, `test_<m>_folding[target|maximize]`,
  `test_<m>_dwc` (markers `fifo_sizing`, `folding`, `dwc`). They resume the convert checkpoint of a
  cached estimate-only build with the committed folding (its estimate = the throughput reference) and
  check stitched-IP rtlsim: golden output and >= 0.98 x the build's own estimate. Each PR branch carries
  only its own tests (dev's models); the integration branch carries all models incl. #1725.
* **Branches** (all local, see PROGRESS.md for push commands):
  * `feature/dev/analytical-fifo-sizing`: + per-model FIFO tests, fix 001 (dynamic MVAU
    characterization: bert), fix 006 (pacer-bound chains + refill floor: balanced foldings).
  * `feature/dev/set-folding-optimizer`: + builder maximize mode (`folding_style="optimizer"` without
    target/JSON), `folding_seed`, platforms KV260_SOM/VCK190/V80, per-model folding tests, **refactored
    `set_folding.py` (2899 -> 1949 lines, behaviour-identical, `agents/folding_refactor/REPORT.md`)**,
    fix 004 (DWC bottleneck in the repair loop, no padding next to a reshape), fix 005 (phantom URAM
    estimate of embedded MVAU weights + maximize search within the as-built budget), fix 003 (see
    PROGRESS.md for whether it landed).
  * `feature/dev/generalized-datawidthconverter`: rebased onto the new folding tip, per-model DWC tests
    (pre-check tries folding seeds 1-8 for a padded folding).
  * **new** `feature/dev/hls-exp-cycles-overhead` (on `upstream/dev`): fix 002, MVAU_hls +7 and
    LabelSelect_hls +4 cycles per frame in `get_exp_cycles` (HLS top function not frame-pipelined).
  * `dev-finnexamples-features`: DWC + FIFO + #1725 + HLS overhead + all tests; since 12:00 fixes were
    cherry-picked (a rebuild would rewrite files under running jobs): rebuild it with
    `tools/rebuild_branches.sh integ` for a clean history before pushing if preferred.
  * `test/{fifo,folding,dwc}`: local test branches of the campaign (`tools/rebuild_branches.sh`).
* **Tooling:** `tools/run_campaign.py` (scheduler: thread/memory budget, watchdog, deadline,
  restartable, single instance via `runner.pid`), `tools/requeue.sh`, `tools/status_table.py`
  (+ `status_notes.json`), `tools/export_results.py`, `tools/build_estimates.py`,
  `tools/rebuild_branches.sh`, `tools/port_feature_tests.py`, `tools/gen_feature_tests.py`,
  `tools/wtenv.sh`, `tools/wtsetup.sh`.
* **Open user decisions:** (1) tfc-w1a1 folding target is infeasible for the optimizer (the committed
  folding is near-fully unrolled and LabelSelect PE=5, the optimizer keeps LabelSelect at PE=1 for
  fmax and FINN's LUT model rejects the unrolled MVAU on ZU3): cap the target or accept. (2) DWC tests
  skip on almost every model: at the reference targets padding does not pay off (vgg10 pads with some
  seeds and passes). (3) upstream bugs found: BIPOLAR weights widened to INT2 by
  `minimize_weight_bit_width` (2x weight memory), FINN's LUT model for binary HLS MVAUs several times
  pessimistic, tiled-MVAU `get_exp_cycles` 4x pessimistic (bert), VVAU_hls has a per-frame HLS
  overhead too; mobilenet maximize folds ~80 nodes at exactly the target period and loses 12 % in
  rtlsim (the optimizer may need a margin).

The sections below are the previous handoff (2026-10-09..19, before the per-model tests).

## 1. What the user asked for (decisions that still hold)

* One `feature/dev/<feature>` branch per feature, each on `upstream/dev` (which contains the
  merged finn-examples PR #1650). They are meant to **replace** `feature/analytical-fifo-sizing`,
  `feature/set-folding-optimizer` and `feature/generalized-datawidthconverter`; the user reviews
  them and decides on the merge. Each branch carries its own unit tests in `tests/` and a
  benchmark test that exercises only its own transformation.
* Benchmark tests check **functional correctness** of each flow and record the **time** of the
  folding and FIFO-sizing transformations. Use the finn-examples helpers, especially model
  caching. Start with small, quickly synthesizable models.
* Target boards: **ZCU104 and U55C** only (real boards the user has; not wired to the agent yet).
* Toolchain: **Vivado/Vitis 2024.2** from `/mnt/labstore/Xilinx` (2023.1 results are stale).
* DWC: use dev's RTL `vpc.sv` and adapt **around** it; never modify its RTL. No HLS generalized
  DWC (it hung in rtlsim).
* DWC padding only makes sense together with folding with `folding_maximum_padding > 0`: it must
  show tighter solutions at a throughput target, or more throughput at full utilization.
* FIFO sizing: never lose more than 1-2 % throughput. Goal: within 1.5x of the theoretical
  optimum depths (**not found in any repo; the user offered to provide them**). Heuristics may be
  removed or simplified if FIFO depth grows by only a few % and throughput never drops.
  Serious algorithm changes are out of scope (no test infrastructure for them).
* Odd folding/FIFO behaviour is fixed by a separate Fable agent with a written manual
  (`agents/<task>/MANUAL.md`), which writes and executes its own plan.
* LabelSelect must stay unfolded: folding its PE ruins fmax (measured, section 5).

## 2. Branches (local and on `origin` = github.com/lstasytis/finn)

| branch | tip | on top of `upstream/dev` (`f2d10d43b`) |
|---|---|---|
| `feature/dev/analytical-fifo-sizing` | `a0ae717a0` | 23 commits: merge of `origin/analytical-fifo-sizing-chained-tav-only`, chained-TAV refactor into named steps, three no-op rules dropped, depth regression test, heuristics docs with measured examples and rule impact, word-based DWC cycles, FIFO benchmark test |
| `feature/dev/set-folding-optimizer` | `7e96ae878` | 15 commits: merge of `origin/feature/set-folding-optimizer`, padding correctness (broadcast thresholds, per-kernel-position conv weights, no XNOR padding), reliability (DWC-aware target, repair/polish, cheapest start, padded seeded by unpadded), FIFO-economical foldings + FIFO-aware mode, LabelSelect kept unfolded, folding benchmark test |
| `feature/dev/generalized-datawidthconverter` | `09882b5db` | **stacked on the folding branch** (+3): DWC pad/crop around `vpc`, builder inserts padding DWCs after `step_minimize_bit_width`, DWC padding benchmark tests |
| `dev-finnexamples-features` | `481cdc6fe` | integration branch = DWC branch + merge of FIFO branch; all sweeps ran on it |
| `claude-tools` | this commit | tooling + this handoff; never part of a PR |

When this was written `origin/feature/dev/*` held the same tips except the FIFO branch, which was
one docs commit (`a0ae717a0`) behind (a fast-forward). Push what is missing:

```
git push origin feature/dev/analytical-fifo-sizing dev-finnexamples-features claude-tools
```

Intermediate branches, fully integrated (only their base commits differ, verified with
`git cherry`), safe to delete: `feature/dev/set-folding-optimizer-fixes`,
`feature/dev/set-folding-optimizer-fifo-aware`, `feature/dev/analytical-fifo-sizing-clarity`,
`dev-finnexamples-features-v3` (backup of an older integration build). The commit hashes inside
`agents/*/REPORT.md` refer to these intermediate branches; the same changes sit on the
`feature/dev/*` branches under new hashes.

### Rebuilding the integration branch

```
git checkout -B dev-finnexamples-features feature/dev/generalized-datawidthconverter
git merge feature/dev/analytical-fifo-sizing
```

Check afterwards (git merges both silently): `streamingdatawidthconverter.py` must contain
exactly one `get_exp_cycles` (the pads-aware, word-based one), and the FIFO tree model method
must start with `if self.pads_or_crops(): return None`. Enable `git config rerere.enabled true`;
`tests/fpgadataflow/test_fpgadataflow_dwc.py` conflicts on every rebuild.

## 3. Setting up a node

* Container: `run-docker.sh` with `FINN_XILINX_PATH=/mnt/labstore/Xilinx
  FINN_XILINX_VERSION=2024.2`; see `RESTART_CONTAINER.md` (restart without reinstalling Claude).
* `tools/env24.sh`: sources 2024.2 in place (strips other `/Xilinx/` entries from PATH and
  LD_LIBRARY_PATH) and sets `PYTHONUSERBASE=/tmp/home_dir/.local` (without it `import finn` fails
  in the agent's shell). Rebuild `finn_xsi/xsi.so` after switching Vivado versions.
* Models: `tests/benchmark/models/download_models.sh`. The BNN-PYNQ models (tfc/cnv) must be
  fresh Brevitas exports (`export_bnn_models.py`); the old 2020 ONNX files break the
  preprocessing merge.
* Extra worktrees need symlinks to the main worktree's `deps/`, `finn_xsi/xsi.so` and
  `tests/benchmark/models/*.onnx` (`/deps` is in `.git/info/exclude`).
* Disk: the build volume (92 GB) filled up twice. HLS projects are large; the harness prunes
  `sol1/.autopilot` of exported IPs, but delete `code_gen_*`, `rtlsim_*`,
  `vivado_stitch_proj_*` of finished runs that the IP cache does not reference (never delete
  directories of builds still running; the agents' cleanup skips anything younger than 3 h).
* Lint: black 23.3.0, isort 5.12.0, ruff (`black --line-length 100`).
* Machine is shared: at most `pytest -n 4`.

## 4. The benchmark feature-flow tests

Harness `tests/benchmark/feature_flow_helpers.py` (byte-identical on all three branches) and
report `tests/benchmark/feature_flow_report.py`; README section "Feature-flow tests" in
`tests/benchmark/README.md`.

* `cached_checkpoint`: builds each model once up to the streamlined/converted phase and reuses
  it (`find_cached_build`); the golden input/output pair comes from the streamline checkpoint.
* `run_flow`: runs the remaining phases with the flow's settings, `verify_step` against the
  golden output (folded cppsim; plus stitched-IP rtlsim in full mode), per-step times, HLS IP
  cache (`HLSIPCache`: content hash of the generated code, keyed on `XILINX_VIVADO`), FIFO and
  DWC stats; appends one JSON row to `$FINN_BUILD_DIR/feature_flow_results/<flow>.jsonl`.
* Env vars: `FEATURE_FLOW_MODELS` / `FEATURE_FLOW_BOARDS` (selection), `FEATURE_FLOW_QUICK=1`
  (no synthesis: cppsim only, minutes), `FEATURE_FLOW_VERIFY`, `FEATURE_FLOW_FRAMES`,
  `FEATURE_FLOW_REBUILD_CACHE`, `FEATURE_FLOW_IP_CACHE`.
* Default models: tfc-w1a1, cnv-w1a1, cybersecurity-mlp; `EXTRA_FEATURE_MODELS` adds tfc-w2a2,
  cnv-w2a2, gtsrb, kws, vgg10-radioml, mobilenet_v1. Avoid cybersecurity-mlp in full mode (HLS
  takes hours).

| test | what it asserts |
|---|---|
| `test_fifo_sizing_flow.py` | `heuristic_analytical` / `heuristic_rtlsim` vs reference `largefifo_rtlsim`: correct, throughput within 2 % |
| `test_folding_flow.py` | naive and optimizer folding at the committed folding JSON's throughput: correct (target recorded, not asserted) |
| `test_dwc_padding_flow.py::test_padding_tightness` | optimizer with/without padding over targets 1-20x the reference cycles plus maximize mode: padded never worse (2 %) |
| `test_dwc_padding_flow.py::test_padding_flow` | a padded folding through the builder, correct in cppsim/rtlsim; skipped if no padding DWC was needed |

Run, e.g.:
```
FEATURE_FLOW_QUICK=1 pytest tests/benchmark/test_folding_flow.py -n 4
FEATURE_FLOW_MODELS=gtsrb,cnv-w2a2 FEATURE_FLOW_BOARDS=ZCU104 pytest tests/benchmark/test_fifo_sizing_flow.py -n 4
python tests/benchmark/feature_flow_report.py
```

## 5. Results (Vivado 2024.2 unless stated; raw data in `results/`)

`results/feature_flow_report.txt` is the report table of the last sweep (latest row per
flow/model/board). Note: its `folding_optimizer`/`dwc_padding` cnv-w1a1 U55C rows (165.8 kbit
FIFOs) predate the FIFO-economical folding fix.

* **All flows functionally correct** on ZCU104 and U55C for every model run (2023.1 and 2024.2).
* **FIFO sizing time**: `heuristic_analytical` 0.1-1.4 s vs `largefifo_rtlsim` 1350-11000 s and
  `heuristic_rtlsim` 2200-11300 s per model. Throughput identical to the reference (ratio 1.000)
  on every simulated model, at 1/2-1/3 the storage (cnv-w1a1 11.9 vs 22.3 kbit, cnv-w2a2 25.5 vs
  83.5, gtsrb 9.9 vs 26.7); exceptions: tfc-w1a1 18.3 vs 9.1 kbit and kws 4.3 vs 0.9 kbit
  (burst floor over-sizes).
* **Rule impact** (full table and per-model numbers: `docs/finn/fifo_sizing_heuristics.rst`
  "Impact of each rule" on the FIFO branch, `agents/fifo_clarity/IMPACT.md`): depth-minimizing
  heavy-duty rules are paced graph input (median +190 % storage without it), slack relaxation,
  throttled cap (MobileNet) and the two-frame split; the burst floor is the one heavy-duty
  throughput protector (without it cnv-w2a2 -46 % fps, gtsrb -17 %). The measured global period
  changes nothing on any model (removal candidate, proposed, not done). MobileNet stitched-IP
  rtlsim gives X outputs on ZCU104 (2023.1 and 2024.2), ResNet50 targets V80: both depth-only.
* **Folding time**: naive ~0.1 s; optimizer 2-30 s (effort 50); padded search ~3x unpadded.
  Optimizer meets every feasible target, padding is never worse, padding helps on TFC/MLP
  (6-28 % fewer LUT at the same target), rarely on the CNNs. FIFO-economical foldings: cnv-w1a1
  36.1 -> 15.5 kbit (naive 14.8), cnv-w2a2 23.3 -> 17.5 kbit (`agents/folding_fix/REPORT.md`,
  `agents/folding_fifo/REPORT.md`).
* **LabelSelect fmax** (`results/labelselect_fmax/results.jsonl`, OOC place & route, xczu7ev,
  5 ns): 10 labels 610 MHz at PE=1, 390 at PE=2, 230 at PE=5; 1000 labels ~530 MHz at PE=1, 140
  at PE=10, 13 at PE=100. Hence `("LabelSelect_hls", "PE")` stays in `DO_NOT_FOLD`.

## 6. Open items

1. **User input needed**: the theoretical-optimum FIFO depths for the 1.5x check.
2. Open questions of the FIFO agent (`agents/fifo_clarity/REPORT.md`, `IMPACT.md`): keep two
   passes (causal + non-causal) or one; whether the cap margin is needed (one ResNet50 edge);
   paced vs saturating source model; idle-window floor on branch edges only; removing the
   measured-global-period rule; kws/tfc over-sizing by the burst floor.
3. Folding objective: the budget-weighted sum values 6 URAM like 13 k LUT, so cnv-w1a1 moves an
   MVAU to LUTRAM (lower sum, higher peak utilization). Is peak utilization or "fit first, then
   LUT" the intended objective? (`agents/folding_fifo/REPORT.md` section 6.)
4. `enable_folding_fifo_heuristic=True` (in-loop analytic sizing) works but is 2.5-3.5x slower
   for the same foldings; stays opt-in.
5. MobileNet stitched-IP rtlsim X outputs on ZCU104 (blocks its throughput numbers).
6. Hardware runs on the real ZCU104/U55C once they are wired up.
7. Porting back: the `feature/dev/*` branches are ready for review; after the user's decision
   they replace the old `feature/*` branches / PRs.

## 7. Files in this directory

* `HANDOFF.md` (this), `RESTART_CONTAINER.md`.
* `agents/<task>/`: manual given to each Fable agent, its plan, its report, logs.
  `fifo_clarity` (chained-TAV refactor, docs, rule impact; `TASK2_IMPACT.md` is the second
  task's manual), `folding_fix` (optimizer reliability + padding), `folding_fifo` (FIFO-aware
  folding).
* `results/`: `*.jsonl` (the build dir's `feature_flow_results/` of the last sweep), `feature_flow_report.txt`,
  `padding_tightness.jsonl`, FIFO ablation depth summaries (`ablation_summary*.txt`),
  `impact.json` (rule impact raw numbers), LabelSelect fmax.
* `tools/`: `env24.sh`, `sweep24.sh` (the full sweep), `labelselect_fmax*.py`,
  `fold_fifo_cmp.py` (quick FIFO-storage comparison of foldings), `full_cnv.py`, and
  `fifo_impact/` (the FIFO agent's ablation and throughput scripts: `make_ablations.py`,
  `run_variant.py`, `throughput.sh`, `assemble_impact.py`, ...). Scripts carry the old
  session's absolute paths (`/tmp/claude-1205/...scratchpad`, `/tmp/finn_dev_lstasytis`);
  adjust before reuse.
