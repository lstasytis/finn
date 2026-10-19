# Issue 004: vgg10-radioml (ZCU104) — optimizer misses the target because of a DWC, and padded folding breaks InsertDWC

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this
file when done (if writing the file is refused, put its full text into your final answer).

**Hard deadline:** the machine reboots at ~20:00 on 2026-10-19 (container clock, `date`). Stop
all your processes by 18:45, commit work in progress (`WIP:` message) and write `REPORT.md` by
19:00 (state, findings, what is left, how to resume).

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, rules: read first)
2. `/home/lstasytis/finn/claude-tools/folding/CONTEXT.md`, `/home/lstasytis/finn/claude-tools/dwc/CONTEXT.md`
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/agents/folding_refactor/REPORT.md`
   (structure of the just-refactored `set_folding.py`, the equivalence dump harness, known quirks)
4. `/home/lstasytis/finn/claude-tools/finn-examples-features/PLAN_per_model_tests.md` sections 2-3

## Symptom A: folding target missed (folding branch)
`test_vgg10radioml_folding[ZCU104-target]` (optimizer at the committed folding's estimated
throughput, 243190 fps = 1028 cycles at 4 ns) fails only the target check: the build's own estimate
is 195312 fps = **1280 cycles**, set by `StreamingDataWidthConverter_rtl_3` (between
`ConvolutionInputGenerator_rtl_4` 770 cycles and `MVAU_rtl_2` 1024 cycles); every compute node is
<= 1088. Output is correct; stitched-IP rtlsim interval is 1088 cycles (1.18x faster than the
estimate). So the optimizer's DWC model and the DWC node's `get_exp_cycles()` (pads-aware,
word-based: max(in, out) words for integer width ratios, in + out words otherwise) disagree. The
refactor report lists a kept quirk: "`compute_hls_dwc_cost` uses the producer's input width as the
DWC's output width"; check whether that (or `dwc_cycles`) is the cause. The HANDOFF says the
optimizer should already be DWC-aware ("DWC-aware target").
* Build: `/home/lstasytis/finn/build/finn_build/fx/folding/fold-target_vgg10-radioml_ZCU104_*/`
  (`report/estimate_layer_cycles.json`, `auto_folding_config.json`, `intermediate_models/`)
* Log: `/home/lstasytis/finn/build/finn_build/fx/logs/campaign/fold-target__vgg10-radioml__ZCU104.log`

## Symptom B: padded folding breaks DWC insertion (DWC branch, maybe folding branch)
With `folding_maximum_padding=4` (the `test_vgg10radioml_dwc` setting) InsertDWC asserts
`Pool_hls_6 -> MVAU_rtl_7: DWC can only pad/crop within a row's last word`: the optimizer picks a
padding the padding DWC cannot realize. Either the optimizer must not propose such paddings (folding
branch), or the DWC must support them (DWC branch, but never modify dev's RTL `vpc.sv`); pick the
minimal, correct fix. The refactor agent's dumps recorded this (`/home/lstasytis/finn/build/finn_build/fx/refactor/dumps/base1/`, vgg10 padded).

## Where to work
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix004 -b fix/004 test/dwc
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix004
cd build/wt/fix004 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
cp -r /home/lstasytis/finn/build/finn_build/fx/dwc/est_vgg10-radioml_ZCU104_* $FINN_BUILD_DIR/   # cached checkpoint
export NUM_DEFAULT_WORKERS=6
python -m pytest -p no:cacheprovider -v "tests/benchmark/vgg10-radioml/test_build_vgg10-radioml.py::test_vgg10radioml_folding[ZCU104-target]"
python -m pytest -p no:cacheprovider -v "tests/benchmark/vgg10-radioml/test_build_vgg10-radioml.py::test_vgg10radioml_dwc[ZCU104]"
```
(each ~15-30 min). `test/dwc` = DWC branch (stacked on the refactored folding branch) + #1725 + FIFO
branch + tests. Another agent (issue 003) is changing `set_folding.py` on `fix/003` for branching
graphs and bert (`_extract_swg_pair`, `_resize_weights`): keep your changes to other functions where
possible, so both cherry-pick cleanly. Fix commits must be self-contained per owning branch
(`feature/dev/set-folding-optimizer` for `set_folding.py`, `feature/dev/generalized-datawidthconverter`
for DWC/InsertDWC code); say in the report which commit goes where.

## Pass criterion
Both vgg10 tests above pass (target met by the estimate; correct rtlsim output; rtlsim >= 0.98 x
estimate). Behaviour gate for `set_folding.py` changes: with the dump harness (refactor report,
"Re-running the gate"), runs that are unaffected by the fix stay byte-identical; explain every
changed run. `tests/fpgadataflow/test_set_folding.py` and `tests/fpgadataflow/test_fpgadataflow_dwc.py`
pass; lint clean.

## Constraints
* Minimal change, match the surrounding style, short comments. Never modify dev's RTL `vpc.sv`. Never use `largefifo_rtlsim`.
* Lint (black 23.3.0 line 100, isort, flake8; `GENERAL.md`). No pushes.
* Resources: at most 16 threads; keep `MemAvailable` above 200 GB; kill only your own PIDs (never
  `pkill -f`). A test campaign and other agents are running: don't touch other worktrees or build
  dirs (reading is fine).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Report (`REPORT.md` next to this file)
Root causes, the fixes (commits on `fix/004`, target branch of each), evidence, open points.
