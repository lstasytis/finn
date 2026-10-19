# Manual: refactor `set_folding.py` (folding optimizer PR) — behaviour-preserving

You are a fresh agent with no prior context. Work autonomously, write your own plan
(`PLAN.md` next to this file), and finish with `REPORT.md` next to this file.

## Hard deadline

The machine reboots at about **20:00 on 2026-10-19** (container time, `date`). Stop all your
processes by **19:00**, commit your work-in-progress on your branch (even if unfinished, with a
`WIP:` commit message) and write `REPORT.md` (state, what is done, what is left, how to resume) by
**19:15**. Nothing that is only in memory or in a running process survives.

## Reading list

1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (repo primer: env,
   worktrees, build dirs, lint, conventions; read first)
2. `/home/lstasytis/finn/claude-tools/README.md`, `/home/lstasytis/finn/claude-tools/folding/CONTEXT.md`
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/HANDOFF.md` (sections 1, 2, 5)
4. `/home/lstasytis/finn/claude-tools/finn-examples-features/agents/folding_fix/REPORT.md` and
   `agents/folding_fifo/REPORT.md` (why the current code looks like it does)

## Setup

```bash
cd /home/lstasytis/finn
git worktree add build/wt/refactor -b refactor/set-folding feature/dev/set-folding-optimizer
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/refactor
cd build/wt/refactor && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
```

Work only in that worktree, on branch `refactor/set-folding`. Never push. Never touch other
worktrees, branches or build directories (a test campaign is running on the machine).
Resources: use at most **16 threads** and keep `MemAvailable` above 200 GB (check
`/proc/meminfo`); never `pkill -f`, stop your own processes by PID.

## Task

Refactor the new code in `src/finn/transformation/fpgadataflow/set_folding.py` (about 2900
lines; upstream dev's version is 326) to far fewer lines: pythonic but readable, with docstrings /
comments on the main functions. Target: **<= ~1500 lines**, self-contained in this one file.

* Keep `apply_naive_folding` (dev's original algorithm, ~241 lines) close to dev's code so the PR
  diff stays readable (`git diff upstream/dev -- src/finn/transformation/fpgadataflow/set_folding.py`).
* Hotspots: `_pair_swg_with_mvau` / `_pair_swg_with_vvau` (93 + 73 lines, near-duplicates),
  `Parameter.update_weight_tensor` / `update_threshold_tensor`, `SetFolding.__init__` (127 lines
  of knob plumbing -> grouped defaults), `Parameter` / `MetaParameter` / `ParameterSet`,
  `Optimizer` (~1160 lines: `optimize`, `cost_model`, `fifo_resources`, `compute_hls_dwc_cost`,
  `cheapest_guess`), the search layer (`_search_folding`, `_record_search_result`,
  `_sweep_and_refine`, `_refine`, `_fold_at`, `_better`), long historical comment blocks,
  dead/unused options.
* Use comprehensions, `dataclass`es, small dispatch tables. No clever one-liners that hurt
  reading. Match FINN's style (see `PYTHON_STYLE_GUIDE.md` in the repo).
* Docstrings / comments for: `SetFolding` (styles, modes), `apply`, the optimizer entry, the cost
  model, the search/refine strategy, padding, the FIFO-aware mode.
* The public interface (`SetFolding.__init__` keyword arguments, as called by
  `step_target_fps_parallelization` in `src/finn/builder/build_dataflow_steps.py`, and by
  `tests/fpgadataflow/test_set_folding.py`) must stay the same. Removing an unused keyword is
  allowed only if no caller in `src/` and `tests/` passes it; list such removals in the report.

## Acceptance gate (hard): identical behaviour

Before touching any code, write a dump script (keep it in
`claude-tools/finn-examples-features/agents/folding_refactor/dump_folding.py` of the main
worktree `/home/lstasytis/finn`) that runs `SetFolding` on the convert-phase checkpoints of all
models/boards and stores, per run, every hw node's folding attributes (`PE`, `SIMD`,
`parallel_window`, `ram_style`, `resType`, `mem_mode`, `runtime_writeable_weights`, padding
attributes, and any other attribute the optimizer sets) plus the cycle estimate
(`dataflow_performance(model)["max_cycles"]`) and the node list, as JSON.

* Checkpoints: `/home/lstasytis/finn/build/finn_build/fx/folding/est_<model>_<board>_*/intermediate_models/phase_convert_to_hardware.onnx`
  (reference estimate builds of the per-model feature tests; being built right now, the
  `report/estimate_network_performance.json` in that dir marks a finished one). The builder's
  own call of SetFolding happens after this checkpoint in `phase_optimize_hardware`
  (`step_target_fps_parallelization`); replicate what that step does before/around SetFolding
  (e.g. the cfg knobs it passes) so the dump exercises the real call. The reference target is the
  `max_cycles` in that build's `report/estimate_network_performance.json`.
* Modes: naive at the reference target; optimizer at the reference target; optimizer maximize
  (`target_cycles_per_frame=None`); padded (optimizer at the reference target with
  `folding_maximum_padding=4`); FIFO-aware (`enable_folding_fifo_heuristic=True`, optimizer at the
  reference target). Use a fixed seed where the code takes one and no search timeout, so runs are
  deterministic; verify determinism by dumping twice on the unmodified code first. If some
  model/mode is too slow (> ~15 min) leave it out and say so.
* Run the dump on the unmodified branch tip, then after every refactor step; the dumps must be
  identical (byte-identical JSON after sorting keys). `tests/fpgadataflow/test_set_folding.py`
  passes (`python -m pytest -q -n 8 tests/fpgadataflow/test_set_folding.py`), lint clean
  (`claude-tools/LINTING.md`: black 23.3.0 `--line-length 100`, isort 5.12.0, flake8 6.0.0; a
  ready env is at `/home/lstasytis/finn/.lintenv`).
* Commit in small steps on `refactor/set-folding` (each commit passing the gate), so a partial
  result is still usable. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Report (`REPORT.md` next to this file)

Before/after line counts (file total and per main class/function), what was merged/removed, the
equivalence evidence (models x modes covered, dump file paths, the diff result), the test and lint
results, commits, open points. Keep it short and factual.
