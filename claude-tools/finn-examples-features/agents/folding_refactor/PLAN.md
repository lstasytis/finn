# Plan: behaviour-preserving refactor of `set_folding.py`

Worktree `/home/lstasytis/finn/build/wt/refactor`, branch `refactor/set-folding` (from
`feature/dev/set-folding-optimizer` @ `1aa33965e`). Deadline: processes stopped 19:00, WIP commit +
REPORT.md by 19:15.

## Equivalence harness

* `dump_folding.py` (this folder) loads a given `set_folding.py` **from a file** and runs
  `SetFolding` like `step_target_fps_parallelization` (cfg defaults, feature-test overrides,
  `apply_to_subgraphs=True`, `seed=0`, no timeout) on every finished
  `fx/folding/est_<model>_<board>_*/intermediate_models/phase_convert_to_hardware.onnx`, target =
  that build's `max_cycles`. Modes: naive, target, maximize, padded (`folding_maximum_padding=4`),
  fifo (`enable_folding_fifo_heuristic=True`). Per run one JSON: all nodes with all attributes,
  tensor shapes/datatypes, initializer hashes, FINNLoop bodies, `dataflow_performance`,
  `last_search_report`. Timings separate (`_timing.json`).
* FINN environment for the dumps: a `git archive` snapshot of `test/folding` @ `12ab5c4cc`
  (`fx/refactor_env`; has the analytic FIFO sizer and the #1725 ops the checkpoints need), so the
  only thing that changes between dumps is the loaded `set_folding.py`. Workers single-threaded,
  pinned to CPUs 48-55,112-119 (16 threads), 15 min timeout per run.
* Dumps in `/home/lstasytis/finn/build/finn_build/fx/refactor/dumps/<label>/`; baseline = the
  branch tip's file (`baseline_set_folding.py`), dumped twice (`base1`, `base2`) to show determinism.
  Gate per step: `diff -r` against `base1` (ignoring `_timing.json`, logs), unit tests
  (`pytest -n 8 tests/fpgadataflow/test_set_folding.py`), lint.

## Refactor steps (one commit each, each passing the gate)

1. Dead code and options: unused attributes/methods (`bound_value_last`, `ParameterSet.filter/get_vals`,
   `MetaParameter.get_cycles`, `allowed_divisors` exponent/skip args, relaxation factor, `verbose`,
   class flags that are constant: `optimize_folding`, `insert_dwcs`, `consider_dwc_costs`,
   `fifo_proxy`), duplicated banners and historical comment blocks.
2. Parameter model: `Parameter` as a dataclass, threshold/weight resize as one table-driven method,
   compact `MetaParameter` / `ParameterSet`.
3. Pairing rules: one SWG helper; `_pair_swg_with_mvau` / `_vvau` / `_pool` share the parameter
   construction.
4. Optimizer: constructor knobs, `optimize` partitioning, `cost_model`, `fifo_resources`,
   `compute_hls_dwc_cost` / `dwc_cycles` shared edge walk, `cheapest_guess`.
5. SetFolding: `__init__` defaults grouped, search layer (`_search_folding`,
   `_record_search_result`, `_sweep_and_refine`, `_refine`, `_fold_at`, `_better`).
6. Docstrings (styles/modes, apply, optimizer entry, cost model, search/refine, padding, FIFO-aware),
   final lint; naive folder kept as close to dev as possible.

Invariants to keep exactly: the order and filters of every `apply_updates` call (they decide which
attributes and tensors change when), `MetaParameter.meta_value` starting at `None`, value sort
order, de-duplication order in the pairing rules, the RNG use (`gen_finn_dt_tensor`, seeded
`dual_annealing`), float summation order in costs.
