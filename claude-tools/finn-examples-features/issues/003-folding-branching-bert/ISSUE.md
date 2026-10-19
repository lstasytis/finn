# Issue 003: folding optimizer fails on branching graphs (resnet18, resnet50, yolov8) and on bert

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this
file when done (if writing the file is refused, put its full text into your final answer).

**Hard deadline:** the machine reboots at ~20:00 on 2026-10-19 (container clock, `date`). Stop
all your processes by 18:45, commit work in progress (`WIP:` message) and write `REPORT.md` by
19:00 (state, findings, what is left, how to resume).

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, rules: read first)
2. `/home/lstasytis/finn/claude-tools/folding/CONTEXT.md`
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/agents/folding_refactor/REPORT.md`
   (the file was just refactored; it describes the structure, the equivalence dump harness and
   the open points this issue is about)
4. `/home/lstasytis/finn/claude-tools/finn-examples-features/PLAN_per_model_tests.md` sections 2-3

## Symptom (found by the refactor agent's dumps; identical before and after the refactor)
`SetFolding(style="optimizer")` (and so `test_<model>_folding[target|maximize]` and
`test_<model>_dwc`) raises on:
1. **resnet18 (4w4a, 8w8a), resnet50, yolov8 (4w4a, 8w8a):** `_extract_swg_pair` treats
   `graph.node[node_index + 1]` as the SWG's consumer. In branching graphs (residual fork/join,
   yolo's concat/split) that is a node of another branch, so the pairing is wrong and the optimizer
   raises. It should use the graph's real consumer (`model.find_consumer`/`find_consumers`).
2. **bert:** `_resize_weights` gets `None` from `get_initializer`: bert's attention MVAUs are
   `mem_mode="dynamic"` (both operands streamed, no weight initializer), and possibly other nodes
   have no initializer either. The optimizer must handle such nodes (e.g. no weight resizing / no
   padding of the dynamic operand; folding constraints that keep both inputs consistent).
3. Minor: yolov8 `naive` style raises a numpy-int `TypeError`; upstream #1725 adds an `int()` cast
   in `common_divisors` (already on the test branches); add the same cast on the folding branch
   only if needed for your tests (identical lines merge cleanly).

Exact errors: run the dump harness for these checkpoints (see the refactor report, section
"Re-running the gate"; dumps of the original file: `/home/lstasytis/finn/build/finn_build/fx/refactor/dumps/base1/`
contain the recorded exceptions per model/mode).

## Where to work
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix003 -b fix/003 test/folding
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix003
cd build/wt/fix003 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
```
`test/folding` = `feature/dev/set-folding-optimizer` (refactored `set_folding.py`) + upstream #1725
(the new models) + the FIFO branch + the per-model feature tests. The fix itself belongs to
`feature/dev/set-folding-optimizer`: keep the fix commits to `src/finn/transformation/fpgadataflow/set_folding.py`
(and, if needed, its unit test `tests/fpgadataflow/test_set_folding.py`), self-contained, so they
cherry-pick cleanly onto that branch. Reference estimate builds (convert checkpoints) for all models:
`/home/lstasytis/finn/build/finn_build/fx/folding/est_<model>_<board>_*/` (copy into your own
`$FINN_BUILD_DIR` to skip rebuilding them; never write into other worktrees' build dirs).

## Pass criterion
1. The dump harness runs the optimizer modes (target, maximize, padded, FIFO-aware) on resnet18
   (both), resnet50, yolov8 (both) and bert without exceptions, and the foldings meet the target
   in target mode where the estimate allows it.
2. **Behaviour gate:** every dump run that did not raise before is byte-identical after your fix
   (compare against `dumps/base1` with the harness; only previously failing runs may change).
3. `tests/fpgadataflow/test_set_folding.py` passes; lint clean.
4. If time allows, run one full feature test, e.g.
   `python -m pytest -p no:cacheprovider -v "tests/benchmark/resnet18/test_build_resnet18.py::test_resnet18_folding[U55C-resnet18_4w4a-target]"`
   (several hours for the big models; report how far it got if it can't finish).

## Constraints
* Minimal change, match the surrounding (refactored) style, short comments. Never modify dev's RTL `vpc.sv`. Never use `largefifo_rtlsim`.
* Lint (black 23.3.0 line 100, isort, flake8; `GENERAL.md`). No pushes.
* Resources: at most 16 threads; keep `MemAvailable` above 200 GB; kill only your own PIDs (never
  `pkill -f`). A test campaign and other fix agents are running: don't touch other worktrees or
  build dirs (reading is fine).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Report (`REPORT.md` next to this file)
Root causes, the fix (commits on `fix/003`), the gate evidence, test results, open points.
