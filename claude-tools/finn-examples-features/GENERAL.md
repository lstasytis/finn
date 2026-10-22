# GENERAL: repo primer for agents working on the finn-examples feature tests

Repo: `/home/lstasytis/finn` (main worktree, stays on branch `claude-tools`, which holds this
folder). FINN dataflow compiler; we work on three PR branches on top of `upstream/dev`
(Xilinx/finn) and test them on the finn-examples benchmark models.

## Branches

| branch | what |
|---|---|
| `feature/dev/analytical-fifo-sizing` | FIFO PR: analytical (tree-model) FIFO sizing, strategy `heuristic_analytical` |
| `feature/dev/set-folding-optimizer` | folding PR: resource-aware folding optimizer (`folding_style="optimizer"`) |
| `feature/dev/generalized-datawidthconverter` | DWC PR, **stacked on the folding branch**: padding/cropping DWCs, `folding_maximum_padding` |
| `dev-finnexamples-features` | integration: DWC + FIFO + upstream PR #1725 (`upstream/update/finn-examples`: resnet18, bert, espcn, yolov8) + per-model feature tests |
| `test/fifo`, `test/folding`, `test/dwc` | local throwaway test branches (feature + #1725 [+ FIFO] + the tests), rebuilt by `tools/rebuild_branches.sh` |
| `claude-tools` | tooling and docs only, never part of a PR |

`origin` = github.com/lstasytis/finn (no ssh; fetch over HTTPS:
`git fetch https://github.com/lstasytis/finn.git '+refs/heads/*:refs/remotes/origin/*'`).
**Never push**; print the `git push` commands for the user. Git identity is repo-local.
`git config rerere.enabled true` is set (merges of these branches conflict in
`src/finn/util/test.py` and `tests/fpgadataflow/test_fpgadataflow_dwc.py`).
Invariant after merging DWC + FIFO (git merges it silently wrong):
`src/finn/custom_op/fpgadataflow/streamingdatawidthconverter.py` has exactly one
`get_exp_cycles` (pads-aware, word-based) and the FIFO tree-model method starts with
`if self.pads_or_crops(): return None`.

## Worktrees and environment

Worktrees live in `/home/lstasytis/finn/build/wt/<name>` (`integ`, `fifo`, `folding`, `dwc`,
`pr-folding` = `feature/dev/set-folding-optimizer`, plus fix/refactor worktrees). A new worktree
needs the untracked bits linked in: `claude-tools/finn-examples-features/tools/wtsetup.sh <wt>`
(deps/, finn_xsi/xsi.so, models, verification goldens).

```bash
cd /home/lstasytis/finn/build/wt/<name>
source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
```

`wtenv.sh` sources Vivado/Vitis 2024.2 (`/mnt/labstore/Xilinx`), sets `PYTHONUSERBASE`
(else `import finn` fails), `FINN_ROOT` and `PYTHONPATH` to the worktree, and
`FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/<name>` (never build in `/tmp`, it has
13 GB). No Xilinx licence for Versal AI Core synthesis: never request synthesis outputs for
VCK190 (bert); rtlsim works.

## Machine rules

128 threads, 755 GB, shared with other users. Our budget is 80 %: at most 102 busy threads, and
`MemAvailable` (`/proc/meminfo`) must stay above ~151 GB. **An OOM crashes the container.**
Never `pkill -f` / `pgrep -f` (it kills your own shell); record PIDs and kill PID trees. Paused
(state `T`) xsim/python processes of an older experiment exist: leave them alone.

## Per-model feature tests

In `tests/benchmark/<model>/test_build_<model>.py` (one block per file, replicated on purpose;
helpers in `tests/benchmark/benchmark_helpers.py`). Markers `fifo_sizing`, `folding`, `dwc`.

* `reference_estimate(board[, model])`: cached estimate-only build with the committed folding JSON
  (`$FINN_BUILD_DIR/est_<model>_<board>_*`, stops after `phase_optimize_hardware`). Its
  `intermediate_models/phase_convert_to_hardware.onnx` is the checkpoint all feature builds resume
  from; its `report/estimate_network_performance.json` `estimated_throughput_fps` is the
  reference throughput.
* `run_feature_build(...)`: resumes the checkpoint with overrides, builds stitched IP, verifies
  with `stitched_ip_rtlsim` (golden I/O of the model), measures rtlsim throughput; asserts
  correct output and `rtlsim fps >= 0.98 x own estimate`. Writes `report/feature_metrics.json`.
* Tests: `test_<m>_fifo_sizing[heuristic_analytical|heuristic_rtlsim]` (committed folding),
  `test_<m>_folding[target|maximize]` (optimizer, FIFOs `heuristic_analytical`; target = reference
  fps, must be met by the estimate), `test_<m>_dwc` (optimizer estimate-only with padding off vs
  `folding_maximum_padding=4`, same seed, at the reference target for seeds 1-8 and, full size, in
  maximize mode; the seed where padding saves the most resources is built; never skipped; fails if
  padding needs > 5 % more resources; numbers in `feature_metrics.json` `pad_compare`, issue 022).
* Never use `largefifo_rtlsim`.
* After a resource-estimate change, move the stale `est_*` dirs aside (`superseded_<issue>/`, once no
  running job uses them; `tools/regen_reference_estimate.py` builds a fresh one that sorts first
  while jobs still use the old). `tools/make_summary_tex.py` takes res./ref. references from
  `tools/ref_resources.py` (FIFO builds re-estimated with the folding worktree's code, issue 023).

Run one test (example):

```bash
cd /home/lstasytis/finn/build/wt/fifo && source .../tools/wtenv.sh . && export NUM_DEFAULT_WORKERS=4
python -m pytest -p no:cacheprovider -v tests/benchmark/gtsrb/test_build_gtsrb.py \
    -m fifo_sizing -k heuristic_analytical
```

Build outputs: `$FINN_BUILD_DIR/<test-tag>_<model>_<board>_<random>/` (`build_dataflow.log`,
`report/`, `verification_output/verify_stitched_ip_rtlsim_0_SUCCESS.npy` or `_FAIL`).
Campaign logs and junit XML: `/home/lstasytis/finn/build/finn_build/fx/logs/`; status table:
`claude-tools/finn-examples-features/STATUS.md`; issues: `claude-tools/finn-examples-features/issues/`.

## Lint and conventions

`/home/lstasytis/finn/.lintenv` (black 23.3.0, isort 5.12.0, flake8 6.0.0):

```bash
LENV=/home/lstasytis/finn/.lintenv
PYTHONPATH=$LENV python3 $LENV/bin/black --line-length=100 <files>
PYTHONPATH=$LENV python3 $LENV/bin/isort <files>
PYTHONPATH=$LENV python3 $LENV/bin/flake8 --max-line-length=100 --extend-ignore=E203 <files>
```

Minimal changes, match the surrounding style, short comments. Never modify dev's RTL `vpc.sv`.
Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
