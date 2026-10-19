# Issue 002: tfc (BNN-PYNQ MLP) — rtlsim throughput ~0.53x of the analytic estimate

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this
file when done.

**Hard deadline:** the machine reboots at ~20:00 on 2026-10-19 (container clock, `date`). Stop
all your processes by 18:45, commit work in progress (`WIP:` message) and write `REPORT.md` by
19:00 (state, findings, what is left, how to resume).

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, rules: read first)
2. `/home/lstasytis/finn/claude-tools/fifo-sizing/CONTEXT.md`
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/HANDOFF.md` sections 1-5
4. `/home/lstasytis/finn/claude-tools/finn-examples-features/PLAN_per_model_tests.md` sections 2-3

## Symptom
The FIFO feature tests (committed folding JSON, FIFOs sized by `heuristic_analytical`) of the
tfc models fail only the throughput check; the stitched-IP rtlsim output is correct:

| test | est. cycles/frame (max node) | rtlsim interval (cycles) | ratio |
|---|---|---|---|
| `test_bnnpynq_fifo_sizing[AUP-ZU3_8GB-tfc-w2a2-heuristic_analytical]` | 8 | 15 | 0.53 |
| `test_bnnpynq_fifo_sizing[AUP-ZU3_8GB-tfc-w1a2-heuristic_analytical]` | 8 | 15 | 0.53 |
| optimizer folding of tfc-w1a1 (integration branch smoke test) | 11 (LabelSelect) | 16 | 0.69 |

All per-node estimates are tiny (2-11 cycles per frame: the committed tfc foldings are almost
fully parallel). The rtlsim number is `stable_throughput[images/s]` of
`report/rtlsim_performance.json` (`rtlsim_batch_size=2`, one steady-state interval).
HANDOFF section 5 reports that the old harness saw the same rtlsim throughput for tfc-w1a1 with
`largefifo_rtlsim` FIFOs as with analytical ones, which points to a fixed per-frame overhead that
`get_exp_cycles()` does not model (e.g. HLS MVAU / LabelSelect / Thresholding pipeline drain or
setup per frame, DWCs) rather than to undersized FIFOs. Verify that, don't assume it.

* Logs: `/home/lstasytis/finn/build/finn_build/fx/logs/campaign/fifo__bnn-pynq__AUP-ZU3_8GB-tfc-w2a2.log` (and `-tfc-w1a2`)
* Builds: `/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_bnn-pynq_tfc-w2a2_AUP-ZU3_8GB_*/`
  (`report/estimate_layer_cycles.json`, `report/rtlsim_performance.json`, `report/feature_metrics.json`,
  `stitched_ip/`, `intermediate_models/`)

## Questions to answer (in this order)
1. Which node(s) / what mechanism sets the 15-cycle interval? Use the existing stitched IP:
   simulate more frames (e.g. 8-16; is the interval still 15?), and for diagnosis only, re-run
   the stitched-IP rtlsim with all FIFOs made much deeper by hand (this is a diagnostic, not a
   sizing method; do NOT use the `largefifo_rtlsim` strategy). If deeper FIFOs don't help, find the
   node whose real per-frame cycle count exceeds its `get_exp_cycles()` (node-by-node rtlsim of
   single nodes, or waveform of the stitched IP with `rtlsim_trace`).
2. Is the gap a measurement artefact of `rtlsim_batch_size=2` (only one interval measured)?
3. Is it caused by one of our PR branches (FIFO sizing, DWC cycles) or by dev's node
   implementations / estimates? The answer decides where a fix belongs:
   * FIFO sizing undersizes -> fix on `feature/dev/analytical-fifo-sizing`.
   * Estimate of a node type wrong (`get_exp_cycles` misses a per-frame overhead) -> say so with
     numbers; propose the minimal fix, but do not change dev's node implementations without
     strong evidence; a fix of an estimate that our folding/FIFO code relies on may go to the
     branch that owns it.
   * Measurement artefact -> propose the test change (e.g. a larger `rtlsim_batch_size` in the
     feature builds; the test code is in `tests/benchmark/benchmark_helpers.py` and each model's
     `run_feature_build`).

## Reproduction
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix002 -b fix/002 test/fifo
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix002
cd build/wt/fix002 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
export NUM_DEFAULT_WORKERS=4
python -m pytest -p no:cacheprovider -v \
  "tests/benchmark/bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_fifo_sizing[AUP-ZU3_8GB-tfc-w2a2-heuristic_analytical]"
```
(about 15-20 min; most of it HLS synthesis of the wide MVAUs). Reuse the existing build dirs
above for analysis where possible.

## Pass criterion
A root cause backed by measurements, and either a fix that makes the tfc FIFO tests pass
(rtlsim >= 0.98 x estimate, correct output) or a clear statement why the bar cannot be met at
these foldings plus the proposed resolution. Don't break other models (cnv, kws pass already).

## Constraints
* Fix commits self-contained (src/ or tests/benchmark/ only, say which branch they belong to) on
  `fix/002` in your own worktree; no pushes.
* Minimal change, match the surrounding style, short comments. Never modify dev's RTL `vpc.sv`.
  Never use `largefifo_rtlsim`.
* Lint (black 23.3.0 line 100, isort, flake8; `GENERAL.md`).
* Resources: at most 16 threads; keep `MemAvailable` above 200 GB; kill only your own PIDs
  (never `pkill -f`). A test campaign is running: don't touch other worktrees or build dirs
  (read-only access to them is fine; copy what you want to modify).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Report (`REPORT.md` next to this file)
Root cause with numbers, the fix or the proposed resolution, evidence, open points.
