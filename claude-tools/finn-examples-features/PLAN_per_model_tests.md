# Plan: per-model feature tests (folding, DWC padding, FIFO sizing) + folding refactor

Written 2026-10-19 by the planning session, after review by the user. You (a fresh agent) execute it.
Read first: `claude-tools/README.md`, `claude-tools/finn-examples-features/HANDOFF.md` (branch
layout, env, previous results), the topic `CONTEXT.md`s (`claude-tools/{folding,dwc,fifo-sizing}/`).
These are all on the `claude-tools` branch; keep a copy of this folder at hand when you switch branches
(e.g. a `git worktree` of `claude-tools`, or read files with `git show claude-tools:<path>`).

## 0. Goals (user's words, condensed)

1. Update `dev-finnexamples-features` with the new models of upstream PR #1725
   (`upstream/update/finn-examples`: resnet18, bert, espcn, yolov8; the PR has no review comments).
   Model and golden files: `models.zip`, `verification_io.zip` in the repo root (provided by the user).
2. Replace the separate feature-flow harness by **additional pytests inside each model's test file**
   (`tests/benchmark/<model>/test_build_<model>.py`). Code replication across models is accepted and
   wanted (clearer than generic scripts). Use the upstream helpers in `tests/benchmark/benchmark_helpers.py`
   (`find_cached_build`, `make_benchmark_cfg`, `get_verify_steps`, `check_build_outputs`) and each
   file's `configure_build(board, output_dir, **overrides)`: each feature test just modifies the cfg
   through overrides. Small generic helpers may be added to `benchmark_helpers.py`.
3. Test folding, DWC (padding) insertion and FIFO sizing on **all** models, using the
   `feature/dev/<feature>` branches. Validation at **rtlsim** level: functional correctness, and
   throughput not below what is expected (1-2 % tolerance only where FIFO sizing is involved).
4. Failures: one fresh agent per issue, no prior context, driven by an `ISSUE.md` per issue plus the
   general docs.
5. Keep a routinely updated **status table** (models x features).
6. Use up to 80 % of the machine (CPU and RAM). Small models (up to 6-8 h) first, large ones after.
7. Refactor the folding PR's transformation (`set_folding.py`) to far fewer lines, pythonic but
   readable, with comments/docstrings on the main functions.

## 1. User decisions (2026-10-19)

* **Branches (1b):** the `feature/dev/*` PR branches stay on `upstream/dev`; they carry tests only for
  models dev already has. All testing happens on throwaway local `test/<feature>` branches that add
  #1725 (and what else is needed, see 3.2). New-model tests are kept durably on
  `dev-finnexamples-features` (which includes #1725).
* **Test variants:** FIFO: `heuristic_analytical` first for every model; `heuristic_rtlsim` only at
  the very end, after everything else is tested. **Do not use `largefifo_rtlsim` at all** (neither as
  reference nor to size FIFOs). Folding: optimizer at the reference target, and maximize mode
  (`target_fps=None`). DWC: skip (`pytest.skip`, recorded as "no padding needed") when no padding DWC
  was inserted.
* **Throughput reference = the analytic estimate**, not an rtlsim reference build:
  `report/estimate_network_performance.json["estimated_throughput_fps"]`
  (= clock / `dataflow_performance(model)["max_cycles"]`, i.e. the slowest node's `get_exp_cycles()`),
  written by `step_generate_estimate_reports`.
* **Tolerances:** rtlsim fps >= 0.98 x estimate where FIFO sizing is in the loop (all feature runs
  here); folding target met exactly by the estimate.
* **Boards:** the upstream boards of each model (see 2.2). Versal parts must be installed first (2.2).
* **Folding refactor:** stays self-contained in `set_folding.py`; ~1500 lines is fine.
* Plan/status files live in `claude-tools/finn-examples-features/`.

**Assumption to confirm with the user before Phase 2 starts:** the folding and DWC tests need FIFO
sizing before rtlsim, and without `largefifo_rtlsim` only the FIFO branch's `heuristic_analytical`
qualifies, so `test/folding` and `test/dwc` also merge `feature/dev/analytical-fifo-sizing` (3.2).

## 2. Facts established during planning

### 2.1 Repo / machine

* Repo `/home/lstasytis/finn` (main worktree). `origin` (git@github.com:lstasytis/finn) is **not
  reachable over ssh**; fetch over HTTPS works:
  `git fetch https://github.com/lstasytis/finn.git '+refs/heads/*:refs/remotes/origin/*'`.
  Pushing needs the user: print the `git push` commands, never push yourself.
* `upstream/update/finn-examples` (tip `64b5667ed`) sits on a newer `upstream/dev` than our branches
  (base `f2d10d43b`; dev has 36 more commits incl. the `MakePynqProject` -> `MakePYNQProject` rename).
  Its `src` changes: `build_dataflow_steps.py` (InferUpsample also for Resize), `set_folding.py`
  (`common_divisors` returns python ints), `qonnx_activation_handlers.py`, `streamline/absorb.py`
  (`preserve_thresh_shape`), `benchmark_helpers.py` (`.pdi` on Versal).
* Branch tips at planning time: `origin/dev-finnexamples-features` `481cdc6fe`,
  `origin/feature/dev/set-folding-optimizer` `7e96ae878`,
  `origin/feature/dev/generalized-datawidthconverter` `09882b5db` (stacked on folding),
  `origin/feature/dev/analytical-fifo-sizing` `a0ae717a0`, `origin/claude-tools` `0bf03ca1f`.
  `set_folding.py` is byte-identical on folding, DWC and integration branches.
* Machine: EPYC 9554P, 128 threads, 755 GB RAM (~100 GB in use by others at planning time).
  80 % budget: **<= 102 busy threads, total RAM in use <= 604 GB** (count other users' usage too; read
  `MemAvailable`). OOM crashes the container (see memory "never-oom-the-machine"): memory gate +
  watchdog are mandatory. Kill only by explicit PID trees, never `pkill -f`.
* A paused (SIGSTOP) bisection of the islands work exists (state `T` processes, xsimk/python). Leave
  them alone.
* `FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build` (708 GB free on the volume; `/tmp` has 13 GB
  - never build there). Use per-worktree subdirectories `.../finn_build/fx/<test-branch>/`.
* Toolchain: Vivado/Vitis HLS 2024.2 (`/mnt/labstore/Xilinx/Vivado/2024.2`). Env script:
  `claude-tools/finn-examples-features/tools/env24.sh` (sets PATH/LD_LIBRARY_PATH and
  `PYTHONUSERBASE=/tmp/home_dir/.local`; without it `import finn` fails). Rebuild `finn_xsi/xsi.so`
  if missing for 2024.2. pytest-xdist 3.2.0 is available.
* No Xilinx licence server is configured (`XILINXD_LICENSE_FILE` unset).

### 2.2 Boards (checked with `get_parts` in Vivado 2024.2)

| board | part | models | 2024.2 |
|---|---|---|---|
| AUP-ZU3_8GB | xczu3eg-sfvc784-2-e | bnn-pynq baseline (tfc-w1a1/w1a2/w2a2, lfc-w1a1/w1a2, cnv-w1a1), gtsrb, kws, cybersecurity-mlp | ok |
| KV260_SOM | xck26-sfvc784-2LV-c | espcn, cnv-w1a2 | ok |
| ZCU104 | xczu7ev-ffvc1156-2-e | vgg10-radioml, mobilenet_v1 | ok |
| U55C | xcu55c-fsvh2892-2L-e | resnet18 4w4a/8w8a, yolov8 4w4a/8w8a, cnv-w2a2, mobilenet_v1 | ok |
| VCK190 | xcvc1902-vsva2197-2MP-e-S | bert | ok for rtlsim (installed 2026-10-19); `synth_design` needs a licence |
| V80 | xcv80-lsva4737-2MHP-e-S | resnet50 | ok, incl. `synth_design` (installed 2026-10-19) |

The user added "Versal AI Core Series" and "Versal HBM Series" to the 2024.2 install on 2026-10-19.
Smoke test (one MVAU_hls with decoupled weights + StreamingFIFO_rtl + TLastMarker: HLS ipgen, stitched
IP, xsim rtlsim, output checked) passed on xcvc1902, xcv80 and xczu7ev; script:
`/tmp/claude-1205/-home-lstasytis-finn/111c65d2-2c9d-488f-9030-5170ad9021fa/scratchpad/versal_smoke/finn_smoke.py`
(needs `FINN_ROOT`/`PYTHONPATH` of a current-branch worktree). OOC `synth_design` works on xcv80 but
fails on xcvc1902 ("A valid license was not found for feature 'Synthesis'"): fine for rtlsim-level
validation, but never request synthesis outputs (OOC synth, `stitched_ip_gen_dcp`, bitfile) for VCK190.
Phase 0 does not need to repeat the check.
Exact board list per model: read each model's `@pytest.mark.parametrize` (bnn-pynq: `BNN_BUILDS`,
resnet18/yolov8: `*_BUILDS`, mobilenet: ZCU104 + U55C). Upstream drops `stitched_ip_rtlsim` for some
cases (mobilenet on ZCU104: X outputs; resnet18 per model via `select_verif_steps`): run those on a
board where it is enabled, or mark the cell `n/a` with the reason.

### 2.3 Builder facts the tests rely on

* Phases: `phase_prepare_model`, `phase_optimize_model`, `phase_convert_to_hardware`,
  `phase_optimize_hardware` (folding config / `target_fps` folding, `step_minimize_bit_width`, padding
  DWC insertion on the DWC branch, `step_set_fifo_depths`, estimates), `phase_build_hardware`,
  `phase_generate_outputs`. Folding, DWC padding and FIFO sizing all happen after the
  `phase_convert_to_hardware` checkpoint.
* `cfg.save_intermediate_models` writes `intermediate_models/<step or phase>.onnx`; the phase-level
  checkpoint `phase_convert_to_hardware.onnx` exists. `start_step` reads from the *same* output dir, so
  instead pass the cached checkpoint path as the model file and an explicit `steps` list (the model's
  steps from `phase_optimize_hardware` on; models with custom step lists: slice their own list).
* An estimate-only build (`generate_outputs=[ESTIMATE_REPORTS]`) skips FIFO sizing when the strategy
  needs synthesis: cheap (minutes).
* `build_dataflow_cfg` returns `-1` on failure instead of raising: always `assert ... == 0`. Upstream
  tests never check verification results: check `verification_output/` for the
  `stitched_ip_rtlsim` SUCCESS file (read `step_*` verify code for the exact naming).
* `report/rtlsim_performance.json` (RTLSIM_PERFORMANCE output) gives the measured throughput (read
  `annotate_rtlsim_performance` for the key name); `time_per_step.json` gives step times.
* Folding cfg knobs on our branches: `folding_style` ("optimizer"/"naive"), `target_fps`,
  `folding_maximum_padding`, `folding_effort`, ...; FIFO: `auto_fifo_depths`, `auto_fifo_strategy`
  (`heuristic_analytical`, `heuristic_rtlsim`). Check `build_dataflow_config.py` on each branch.

## 3. Phases

### Phase 0 - environment (first)

1. `source claude-tools/finn-examples-features/tools/env24.sh` (fix stale paths in it), export
   `FINN_BUILD_DIR` as above, `git config rerere.enabled true`.
2. Versal parts: done (2.2).
3. Unzip into fresh dirs, then move: `models.zip` -> `tests/benchmark/models/*.onnx`,
   `verification_io.zip` -> `tests/benchmark/verification_io/*.npy` (both gitignored). Worktrees get
   symlinks to these, to `deps/` and `finn_xsi/xsi.so` (HANDOFF section 3).
4. Write `claude-tools/finn-examples-features/GENERAL.md`: a 1-2 page repo primer for fix agents
   (branches/worktrees, env, how to run one model's feature test, where logs/results are, lint,
   conventions). Keep it current.

### Phase 1 - branches

1. `dev-finnexamples-features`: merge `upstream/update/finn-examples`. Expected conflicts:
   `set_folding.py::common_divisors` (keep ours + the `int()` cast), `build_dataflow_steps.py`,
   `tests/fpgadataflow/test_fpgadataflow_dwc.py` (known), PYNQ renames in our files. Afterwards check the
   HANDOFF invariants: exactly one `get_exp_cycles` in `streamingdatawidthconverter.py` (pads-aware,
   word-based) and the FIFO tree-model method starting with `if self.pads_or_crops(): return None`.
   Run `tests/fpgadataflow/test_set_folding.py`, `test_fifosizing*.py`, `test_fpgadataflow_dwc.py` (quick).
2. Test branches (local only, rebuildable, one worktree each under `/home/lstasytis/finn/build/wt/`):
   * `test/fifo` = `feature/dev/analytical-fifo-sizing` + merge `upstream/update/finn-examples`
   * `test/folding` = `feature/dev/set-folding-optimizer` + #1725 + merge FIFO branch (assumption in 1)
   * `test/dwc` = `feature/dev/generalized-datawidthconverter` + #1725 + merge FIFO branch
   Write the recipe into `GENERAL.md` (they get rebuilt after fixes/refactor).

### Phase 2 - per-model tests (author on `dev-finnexamples-features`, then port)

Per model file, add (replicated per model; names illustrative):

```python
FEATURE_OUTPUTS = [ESTIMATE_REPORTS, STITCHED_IP, RTLSIM_PERFORMANCE]   # no bitfile

def reference_estimate(board):
    # committed folding JSON, estimate only (no FIFO sizing, no HLS): convert checkpoint + target fps
    out = find_cached_build(f"est_<model>_{board}_")
    if out is None or not <estimate_network_performance.json exists>:
        out = make_build_dir(f"est_<model>_{board}_")
        assert build.build_dataflow_cfg(model_file, configure_build(
            board, out, generate_outputs=[ESTIMATE_REPORTS], save_intermediate_models=True)) == 0
    return out

def run_feature(board, tag, **overrides):
    ref = reference_estimate(board)
    out = make_build_dir(f"{tag}_<model>_{board}_")
    cfg = configure_build(board, out, steps=<steps from phase_optimize_hardware>,
                          generate_outputs=FEATURE_OUTPUTS, verify_steps=["stitched_ip_rtlsim"],
                          **overrides)
    assert build.build_dataflow_cfg(ref + "/intermediate_models/phase_convert_to_hardware.onnx", cfg) == 0
    <assert stitched_ip_rtlsim verification SUCCESS>
    return ref, out
```

Tests (markers `fifo_sizing`, `folding`, `dwc`, registered in `setup.cfg`; plus `slow`, `vivado`,
`finn_examples`; `@pytest.mark.xdist_group("<model>")` so one model's tests share the cached estimate
build and run serially while models run in parallel with `-n K --dist loadgroup`):

| test | overrides | asserts |
|---|---|---|
| `test_<model>_fifo_sizing[heuristic_analytical]` | committed folding JSON, `auto_fifo_depths=True, auto_fifo_strategy="heuristic_analytical"` | rtlsim correct; rtlsim fps >= 0.98 x own `estimated_throughput_fps` |
| `test_<model>_fifo_sizing[heuristic_rtlsim]` | same, `heuristic_rtlsim` | same (run only at the very end of the campaign) |
| `test_<model>_folding[target]` | no folding JSON, `folding_style="optimizer"`, `target_fps` = reference estimate fps, FIFO `heuristic_analytical` | own estimate fps >= target; rtlsim correct; rtlsim fps >= 0.98 x own estimate |
| `test_<model>_folding[maximize]` | as above with `target_fps=None` | rtlsim correct; rtlsim >= 0.98 x own estimate; record estimate ratio vs reference (warn, don't fail, if < 1) |
| `test_<model>_dwc` | optimizer at the reference target with `folding_maximum_padding > 0` | skip if no padding DWC was inserted; else as folding[target] |

Record per test (as JSON next to the build and for the status table): estimate fps, rtlsim fps,
ratio, FIFO total kbit, padding DWC count, folding/FIFO step times from `time_per_step.json`.
Allowed additions to `benchmark_helpers.py` (keep byte-identical on all branches to avoid conflicts):
e.g. `rtlsim_throughput(output_dir)`, `estimated_throughput(output_dir)`,
`verification_passed(output_dir, step)`. Everything else stays in the model files.

Then:
1. Delete the old harness on all branches: `tests/benchmark/feature_flow_helpers.py`,
   `feature_flow_report.py`, `test_fifo_sizing_flow.py`, `test_folding_flow.py`,
   `test_dwc_padding_flow.py`, and the README "Feature-flow tests" section (replace with a short
   section describing the per-model feature tests and markers).
2. Smoke test on `tfc-w1a1` (all tests) on the integration branch.
3. Port: each `feature/dev/*` branch gets only its own tests and only for dev's models
   (FIFO -> fifo tests; folding -> folding tests; DWC -> dwc tests); `dev-finnexamples-features` and the
   `test/*` branches carry all models. Keep `set_folding.py` changes off the FIFO branch, etc.
4. Lint (black 23.3.0 `--line-length 100`, isort 5.12.0, flake8 6.0.0 / ruff; `claude-tools/LINTING.md`).
5. Commit locally; print push commands for the user.

### Phase 3 - folding refactor (parallel to Phases 2/4; fresh agent with a manual)

Write `claude-tools/finn-examples-features/agents/folding_refactor/MANUAL.md` and start a fresh agent on
it (worktree of `feature/dev/set-folding-optimizer`). Content of the manual:
* Scope: the new code in `src/finn/transformation/fpgadataflow/set_folding.py` (2899 lines; dev's is
  326). Keep `apply_naive_folding` (dev's original algorithm, ~241 lines) close to dev for a readable PR
  diff. Self-contained in the file, target <= ~1500 lines.
* Hotspots: `_pair_swg_with_mvau` / `_pair_swg_with_vvau` (93 + 73 lines, near-duplicates),
  `Parameter.update_weight_tensor` / `update_threshold_tensor`, `SetFolding.__init__` (127 lines of
  knob plumbing -> grouped defaults), `Parameter`/`MetaParameter`/`ParameterSet`, `Optimizer`
  (~1160 lines: `optimize`, `cost_model`, `fifo_resources`, `compute_hls_dwc_cost`,
  `cheapest_guess`), the search layer (`_search_folding`, `_record_search_result`, `_sweep_and_refine`,
  `_refine`, `_fold_at`, `_better`), long historical comment blocks, dead/unused options.
  Use comprehensions, `dataclass`es, small dispatch tables; no clever one-liners that hurt reading.
* Docstrings/comments for the main functions: `SetFolding` (styles, modes), `apply`, the optimizer
  entry, the cost model, the search/refine strategy, padding, FIFO-aware mode.
* **Acceptance gate (hard): identical behaviour.** Before touching code, write a dump script that runs
  `SetFolding` on the convert checkpoints of all models/boards (from the Phase 2 estimate builds or
  the old cached builds in `FINN_BUILD_DIR`) in modes {naive, optimizer at reference target, maximize,
  padded (`folding_maximum_padding>0`), FIFO-aware (`enable_folding_fifo_heuristic=True`)} with a fixed
  seed and no search timeout, and stores every hw node's folding attributes (PE, SIMD, parallel_window,
  ram_style, resType, mem_mode, padding attrs) + the cycle estimate. After the refactor the dump must be
  identical. `tests/fpgadataflow/test_set_folding.py` passes, lint clean. Report: before/after line
  counts, what was merged/removed, the equivalence evidence.
* Afterwards (orchestrator): rebase the DWC branch on the refactored folding branch, rebuild
  `dev-finnexamples-features` (HANDOFF recipe + merge #1725) and the `test/*` branches. Folding results
  obtained before the refactor stay valid by the equivalence proof.

### Phase 4 - test campaign

* Scheduler script `claude-tools/finn-examples-features/tools/run_campaign.py` (tooling, not PR):
  queue of (test branch worktree, model, board, marker[, param]); launches
  `pytest tests/benchmark/<model>/... -m <marker> -k <board> -n <k> --dist loadgroup --junitxml=...`
  per job with `NUM_DEFAULT_WORKERS` set; starts a job only if busy threads stay <= 102 and
  `MemAvailable` minus the job's memory estimate stays >= 151 GB (20 %). Watchdog: below 20 % free stop
  launching; below 10 % kill the newest job's process tree by PID. Per-job logs under
  `$FINN_BUILD_DIR/fx/logs/`. Restartable (skips jobs with a passing result).
* Order: tier 1 (small; anything whose run exceeds ~8 h moves to tier 2): tfc-w1a1, tfc-w1a2, tfc-w2a2,
  lfc-w1a1, lfc-w1a2, cnv-w1a1, cnv-w1a2, cnv-w2a2, gtsrb, kws, cybersecurity-mlp, vgg10-radioml,
  espcn. Tier 2: mobilenet_v1, resnet18 x2, yolov8 x2, bert, resnet50. Within each tier: FIFO
  analytical, folding (target, maximize), DWC. `heuristic_rtlsim` FIFO tests for all models last.
* Disk: prune finished runs' HLS `.autopilot`, `code_gen_*`/`rtlsim_*` trees not needed for
  debugging (never touch directories of running jobs; keep `est_*` builds and failing runs).
* Step times are recorded but measured under load (not comparable to isolated timings; say so).

### Phase 5 - fix agents

For each failing (model, feature): write `claude-tools/finn-examples-features/issues/<NNN-model-feature>/ISSUE.md`
with: symptom (error excerpt, numbers), log/build paths, the one-line reproduction (worktree, env,
pytest command), the expected behaviour and pass criterion, constraints (minimal change, match style,
never modify dev's `vpc.sv`, no `largefifo_rtlsim`, which `feature/dev/*` branch the fix belongs to, lint,
no pushes), reading list (`GENERAL.md`, `claude-tools/README.md`, `HANDOFF.md`, the feature's
`CONTEXT.md`, this plan's sections 2-3). Start a fresh agent (no context) with only "Follow
`<path>/ISSUE.md`"; it works in its own worktree on `fix/<NNN>` and writes `REPORT.md` next to the
ISSUE. Max 3 fix agents at once, inside the resource budget. The orchestrator reviews, cherry-picks the
fix onto the right `feature/dev/*` branch, rebuilds the `test/*` branch, re-runs the test, updates the
table.

### Phase 6 - status table and delivery

* `claude-tools/finn-examples-features/STATUS.md`, regenerated by
  `tools/status_table.py` after every finished job (from junit XML + the per-test JSON), plus a
  hand-written notes column. Rows: model (+board); columns: estimate fps (committed folding),
  FIFO analytical, Folding target, Folding maximize, DWC, FIFO rtlsim, Notes. Cells: `✅ 0.99x 12.3 kbit 1.2 s`,
  `❌ #NNN`, `⏳`, `⏸`, `skip (no padding)`, `n/a (reason)`.
* At the end: update `HANDOFF.md` (new state, results summary), keep `STATUS.md` final, print the push
  commands (`feature/dev/*`, `dev-finnexamples-features`, `claude-tools`).
