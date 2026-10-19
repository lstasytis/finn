# Folding refactor: report (written by the orchestrator from the agent's hand-back; the agent
# could not write files at the end)

Result: `set_folding.py` 2899 -> 1949 lines, behaviour byte-identical on every dump; target
~1500 not met (236 lines are dev's naive folder, 68 licence/imports, ~410 docstrings/comments as
asked; the remaining ~1230 code lines carry the features; 1500 would need dropping features or
docstrings). Branch `refactor/set-folding` (worktree `build/wt/refactor`), 4 commits, only
`set_folding.py` changed. Rebased onto `feature/dev/set-folding-optimizer` and fast-forwarded
into it by the orchestrator (tip 4b8b7404c at 12:30); the DWC branch was rebased on top.

| commit (after rebase) | what |
|---|---|
| dba79845e | restructure: `Parameter` dataclass, table-driven weight/threshold resizing, slimmer `MetaParameter`/`ParameterSet`, constant `Optimizer` knobs, `_partitions`, shared SWG params, dead flags and historical comments removed |
| b4a15cf2e | `SetFolding`/`Optimizer` as dataclasses (kwargs = documented fields), compact pairing rules, as-built verdict merged into `_record_search_result`, optimizer-flow overview docstring |
| 8dd11c9df | tidy throughput search and cheapest start |
| 4b8b7404c | `Optimizer` fields first, commented |

## Line counts (before -> after)
file 2899 -> 1949; `Parameter` 172 -> 92; `MetaParameter`/`ParameterSet`/`GridEncoder` 79/47/32 ->
24/34/24; `Optimizer` 1161 -> ~680 (`optimize` 123 -> 42 + `_partitions` 19, `cost_model` 65 -> 22,
`compute_hls_dwc_cost`/`dwc_cycles`/`fifo_resources` 60/37/70 -> 33/32/65, `cheapest_guess` 45);
`_pair_swg_with_mvau`/`_vvau`/`_pool` 92/71/23 -> 47/27/10; `SetFolding` 1193 -> 900
(`__init__` 125 -> ~40 dataclass fields + 21 `__post_init__`; `_search_folding` 116 -> 65,
`_sweep_and_refine` 45 -> 32, `_better` 31 -> 26, `_fold_at` 51 -> 28, `_refine` 83 -> 78,
`_record_search_result` 73 -> 66); `apply_naive_folding` (dev) 240 -> 236 (diff vs dev unchanged);
`insert_and_size_fifos` 52 -> 43. `git diff upstream/dev` of the file: +1665/-42 (was +2651/-42).

## Removed / merged
* `Parameter.update_weight_tensor`/`update_threshold_tensor` -> `_resize_weights`/`_resize_thresholds` (table `THRESHOLD_INPUTS`).
* Unused members of `MetaParameter`/`ParameterSet` (`model`, `updated`, `index`, `get_cycles`, `filter`, `get_vals`).
* Pairing rules share `_param`/`_swg_params`; `_extract_swg_pair` uses a dispatch table; `allowed_divisors` lost unused args.
* `Optimizer` knobs that never varied became constants (`hard_constraint_target`, relaxation 1.0, `parameters_to_apply`, `infeasible_penalty`, `maxiter`/`accept`/`visit`, `fifo_proxy`, `verbose`); `cost_model(values)` without the self-passing arg.
* `SetFolding` constant class flags and dead branches (`optimize_folding`, `optimize_resource_types`, `insert_dwcs` incl. the naive folder's no-op `InsertDWC`, `consider_dwc_costs`, `fifo_proxy`).
* **Removed `SetFolding` keywords** (no caller in `src/`/`tests/` passes them): `verbose`, `maxiter`, `accept`, `visit`. All other keywords/defaults and the positional first argument unchanged. Internal attribute/`Optimizer` keyword renames (e.g. `maxfun_per_parameter` -> `effort`).

## Equivalence evidence
`dump_folding.py` (+ `summarize_dump.py`, `run_dump.sh` in the dumps dir) loads a given
`set_folding.py` and calls `SetFolding` like `step_target_fps_parallelization` (cfg defaults + the
feature tests' overrides, `apply_to_subgraphs=True`, `seed=0`, no timeout, target = each estimate
build's `max_cycles`); stores every node with all attributes, shapes, datatypes, initializer hashes,
FINNLoop bodies, `dataflow_performance`, `last_search_report`; exceptions recorded as results.
All dumps on one frozen snapshot of the integration tree (`fx/refactor_env_integ`, @4966c56eb).
22 checkpoints x 5 modes (naive, target, maximize, padded=4, FIFO-aware) = 110 runs, 108 compared
(mobilenet_v1 maximize and FIFO-aware > 15 min, left out); plus 50 runs on 10 fast checkpoints for
`folding_max_attempts=6`, `prefer_memory="uram"`+`prefer_compute="dsp"`, `allow_uram_weights`,
`folding_pad_io_nodes`, `strict_budget`. 33 of the 108 baseline runs end in an exception (compared
by type and message). Determinism: base1 == base2, basex1 == basex2. After every step `diff -r`
vs baseline is empty (step1-4 on the 108, step3x/4x on the 50). Runtime unchanged within noise.
Dumps: `/home/lstasytis/finn/build/finn_build/fx/refactor/dumps/`.

Tests: `tests/fpgadataflow/test_set_folding.py` 29 passed, 5 skipped (need DWC/FIFO), 1 xfail in the
branch worktree; 34 passed, 1 xfail on the integration tree. black/isort/flake8 clean.

## Open points found by the dumps (fail identically on the original file)
* KV260_SOM: `KeyError` in `_resource_budgets` (missing platform) -> fixed by the orchestrator
  (commit "Platforms: KV260_SOM, VCK190 and V80 resource counts").
* resnet18, resnet50, yolov8: `_extract_swg_pair` takes `graph.node[node_index + 1]` as the SWG's
  consumer; in branching graphs that is another branch's node, so the optimizer raises
  (should use `find_consumer`) -> issue 003.
* bert: `_resize_weights` gets `None` from `get_initializer` (MVAU weights not an initializer,
  dynamic MVAUs) -> issue 003.
* vgg10 padded: InsertDWC asserts "Pool_hls_6 -> MVAU_rtl_7: DWC can only pad/crop within a row's last word" -> issue 004.
* yolov8 naive: numpy ints from `common_divisors` (`TypeError`); the #1725 `int()` cast is not on the folding branch.
* Kept, documented quirks: `compute_hls_dwc_cost` uses the producer's input width as the DWC's
  output width; `_partitions` collects nothing when `max_nodes=1` and the first target parameter is
  not on node 0.

## Re-running the gate
```bash
D=/home/lstasytis/finn/build/finn_build/fx/refactor/dumps
cp <worktree>/src/finn/transformation/fpgadataflow/set_folding.py $D/new_set_folding.py
$D/run_dump.sh $D/new_set_folding.py $D/new --jobs 14 --runs-from $D/base1 --exclude "mobilenet_v1:(maximize|fifo)"
diff -r -x '_*' -x '*.log' -x 'mobilenet_v1__U55C__maximize.json' -x 'mobilenet_v1__U55C__fifo.json' $D/base1 $D/new
$D/run_dump.sh $D/new_set_folding.py $D/newx --jobs 14 --runs-from $D/basex1 --modes attempts,prefer,uram,padio,strict
diff -r -x '_*' -x '*.log' $D/basex1 $D/newx
```
(~10 + 3 min; pins workers to CPUs 48-55,112-119.) Note: the gate snapshot predates the
platforms fix, so KV260 runs differ from the baseline after it (expected).
