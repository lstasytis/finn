# Issue 001: bert (VCK190) — `heuristic_analytical` FIFO sizing aborts with an rtlsim timeout

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this
file when done.

**Hard deadline:** the machine reboots at ~20:00 on 2026-10-19 (container clock, `date`). Stop
all your processes by 18:45, commit work in progress (`WIP:` message) and write `REPORT.md` by
19:00 (state, findings, what is left, how to resume).

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, rules: read first)
2. `/home/lstasytis/finn/claude-tools/fifo-sizing/CONTEXT.md`
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/HANDOFF.md` sections 1-4
4. `/home/lstasytis/finn/claude-tools/finn-examples-features/PLAN_per_model_tests.md` sections 2-3

## Symptom
`test_bert_fifo_sizing[VCK190-heuristic_analytical]` on `test/fifo` fails ~30 s into
`step_set_fifo_depths` (in `phase_optimize_hardware`):

```
derive_characteristic.py:216 applyNodeLocal -> matrixvectoractivation.py:1656 derive_token_access_vectors
-> hwcustomop.py:353 derive_token_access_vectors_using_rtlsim -> rtlsim_multi_io
AssertionError: RTL simulation timed out after 98304 cycles (derived estimate: 98304). ...
Triggered watchdogs: [out0_V timeout].
```
So `heuristic_analytical` falls back to per-node rtlsim characterization for a node that has no
tree model, and that simulation never produces output. Suspects: the two **dynamic MVAUs**
(`MVAU_rtl_3`: MW=32 MH=128 PE=8 SIMD=8, `MVAU_rtl_4`: MW=128 MH=32, `mem_mode="dynamic"`, i.e.
attention matmuls with two streamed inputs: the characterization probably drives only one input
or the wrong amount of data on the second), or another node type without a tree model
(bert also has `ElementwiseAdd_rtl`, `ElementwiseMul_rtl`, `OuterShuffle_hls`, `InnerShuffle_rtl`,
`DuplicateStreams_hls`, `LayerNorm_rtl`, `HWSoftmax_rtl`, `SelectToken_rtl`). First find out which
node it is.

* Log: `/home/lstasytis/finn/build/finn_build/fx/logs/campaign/fifo__bert__VCK190.log`
* Build: `/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_bert_VCK190_zgp5j16t/`
  (`intermediate_models/step_transpose_decomposition.onnx` = the model given to FIFO sizing)
* Reference estimate build (cached checkpoint): `/home/lstasytis/finn/build/finn_build/fx/fifo/est_bert_VCK190_*/`

## Reproduction
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix001 -b fix/001 test/fifo
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix001
cd build/wt/fix001 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
# optional: reuse the cached estimate build instead of rebuilding it (a few minutes)
cp -r /home/lstasytis/finn/build/finn_build/fx/fifo/est_bert_VCK190_* $FINN_BUILD_DIR/
export NUM_DEFAULT_WORKERS=8
python -m pytest -p no:cacheprovider -v \
  "tests/benchmark/bert/test_build_bert.py::test_bert_fifo_sizing[VCK190-heuristic_analytical]"
```
A faster loop: load `step_transpose_decomposition.onnx` from the build above and run the
failing transformation (or the single node's `derive_token_access_vectors`) directly.

## Expected behaviour / pass criterion
`heuristic_analytical` sizes bert's FIFOs; the test above passes: stitched-IP rtlsim output
equals the golden output and the rtlsim throughput is >= 0.98 x the build's own estimate
(the full test builds HLS/RTL IP and runs stitched-IP rtlsim; expect 1-3 h). If the full test
cannot finish before the deadline, report how far it got and the evidence for the fix (e.g. the
node now characterizes / gets a tree model, FIFO sizing completes).
Also run `tests/fpgadataflow/test_fifosizing*.py` (`-m "not slow"` subset if long) for regressions.

## Constraints
* The fix belongs to `feature/dev/analytical-fifo-sizing` (the FIFO PR): change only what that
  branch owns (`src/` FIFO sizing / characterization / tree models, its unit tests). Keep the fix
  commits self-contained (no `tests/benchmark` changes, nothing from #1725) so they cherry-pick
  cleanly onto `feature/dev/analytical-fifo-sizing`; work on `fix/001` in your own worktree.
* Minimal change, match the surrounding style, short comments. Never modify dev's RTL `vpc.sv`.
* Never use `largefifo_rtlsim`.
* Lint (black 23.3.0 line 100, isort, flake8; `GENERAL.md`). No pushes.
* Resources: at most 16 threads; keep `MemAvailable` above 200 GB; kill only your own PIDs
  (never `pkill -f`). A test campaign is running on the machine: don't touch other worktrees or
  build dirs.
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Report (`REPORT.md` next to this file)
Root cause, the fix (commits on `fix/001`, files), evidence (test results, numbers), open points.
