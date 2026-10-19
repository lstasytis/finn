# Issue 005: maximize-mode folding on AUP-ZU3_8GB is far slower than the committed folding and over budget

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this
file when done (if writing the file is refused, put its full text into your final answer).

**Hard deadline:** the machine reboots at ~20:00 on 2026-10-19 (container clock, `date`). Stop
all your processes by 18:45, commit work in progress (`WIP:` message) and write `REPORT.md` by
19:00 (state, findings, what is left, how to resume).

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (read first)
2. `/home/lstasytis/finn/claude-tools/folding/CONTEXT.md`, `/home/lstasytis/finn/claude-tools/finn-examples-features/HANDOFF.md`
   (section 6 item 3: the folding objective question), `agents/folding_fix/REPORT.md`, `agents/folding_fifo/REPORT.md`
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/agents/folding_refactor/REPORT.md`
   (structure of the refactored `set_folding.py`, the equivalence dump harness)

## Symptom
`test_<model>_folding[maximize]` = builder with `folding_style="optimizer"`, no `target_fps`, no
folding JSON (new: maximize throughput within the board's resource budget; `folding_seed=1`).
The tests pass (correct rtlsim output; maximize only warns when slower than the committed folding),
but on AUP-ZU3_8GB (xczu3eg: 70560 LUT, 432 BRAM18, 0 URAM, 360 DSP) the result is poor:

| model | committed est. fps | maximize est. fps | ratio | maximize est. LUT / BRAM18 / URAM (FINN estimate reports) | committed |
|---|---|---|---|---|---|
| cnv-w1a1 | 12244 | 439 | 0.036 | 119340 / 3 / 0 | 158669 / 81 / 15 |
| cnv-w1a2 | 3028 | 219 | 0.072 | (similar) | |
| cnv-w2a2 | 3028 | 219 | 0.072 | 227465 / 2 / 1 | 46178 / 178 / 45 |
| lfc-w1a1 | 195312 | 127551 | 0.653 | 201886 / 0 / 0 | 40992 / 158 / 80 |
| cybersecurity-mlp | 1e8 | 6.7e6 | 0.067 | | |
| gtsrb | 8138 | 7812 | 0.960 | | |
| kws | 25510 | 204082 | 8.0 | | |

On KV260_SOM (cnv-w1a2) maximize gives 2.63x the committed throughput, so the mode works on a
larger part. On ZU3 the maximize foldings use 2-4x the LUTs of the whole device by FINN's
estimates with almost no BRAM (all MVAUs `ram_style="auto"`, `resType="lut"`; cnv-w2a2 MVAU
PE/SIMD: (8,9),(2,144),(2,288),(4,288),(8,288),(16,288),(4,256),(4,512),(5,8)), and no "fits the
budget" verdict appears in the build log. Questions: does the optimizer's own resource view disagree
with FINN's estimate reports (e.g. LUTRAM weight memories, `ram_style="auto"` resolution, URAM on a
part without URAM)? Why does the maximize search end at such slow foldings (the committed foldings
are faster AND use fewer LUTs)? Is the budget check / verdict missing in maximize mode?

* Builds: `/home/lstasytis/finn/build/finn_build/fx/folding/fold-max_bnn-pynq_<model>_AUP-ZU3_8GB_*/`
  (`auto_folding_config.json`, `report/estimate_layer_resources.json`, `report/estimate_layer_cycles.json`,
  `build_dataflow.log`); reference estimate builds `.../folding/est_bnn-pynq_<model>_AUP-ZU3_8GB_*/`.
* Logs: `/home/lstasytis/finn/build/finn_build/fx/logs/campaign/fold-max__bnn-pynq__AUP-ZU3_8GB-<model>.log`

## Where to work
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix005 -b fix/005 test/folding
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix005
cd build/wt/fix005 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
```
Fast loop: run `SetFolding(None, style="optimizer", platform="AUP-ZU3_8GB", seed=1, ...)` (as
`step_target_fps_parallelization` calls it) on the convert checkpoint
`.../est_bnn-pynq_cnv-w1a1_AUP-ZU3_8GB_*/intermediate_models/phase_convert_to_hardware.onnx` and
compare its resource view and the estimate reports. Full test (~20 min):
`python -m pytest -p no:cacheprovider -v "tests/benchmark/bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_folding[AUP-ZU3_8GB-cnv-w1a1-maximize]"`.
Other agents change `set_folding.py` on `fix/003` (`_extract_swg_pair`, `_resize_weights`: branching
graphs, bert) and `fix/004` (DWC cost model / padding vs InsertDWC): keep your change to other
functions where possible and self-contained (it belongs to `feature/dev/set-folding-optimizer`).

## Pass criterion
Root cause with numbers. A fix such that maximize on AUP-ZU3_8GB reaches at least the committed
folding's throughput where that folding fits the budget by the optimizer's (corrected) resource
view, and stays within the budget; maximize on other boards not worse. Behaviour gate (dump harness,
refactor report): target-mode runs unchanged unless the fix is in shared code, explain every changed
run. `tests/fpgadataflow/test_set_folding.py` passes; lint clean. If the right fix is an objective
decision (HANDOFF open item 3), implement nothing risky: report the analysis and the options.

## Constraints
* Minimal change, match the surrounding style, short comments. Never use `largefifo_rtlsim`. No pushes.
* Lint (black 23.3.0 line 100, isort, flake8). Resources: at most 16 threads; keep `MemAvailable`
  above 200 GB; kill only your own PIDs (never `pkill -f`). A test campaign and other agents are
  running: don't touch other worktrees or build dirs (reading is fine).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
