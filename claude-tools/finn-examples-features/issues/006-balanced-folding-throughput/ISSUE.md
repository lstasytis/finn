# Issue 006: cnv with balanced (optimizer) foldings — rtlsim 2-5 % below the estimate

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this
file when done (if writing the file is refused, put its full text into your final answer).

**Hard deadline:** the machine reboots at ~20:00 on 2026-10-19 (container clock, `date`). Stop
all your processes by 18:45, commit work in progress (`WIP:` message) and write `REPORT.md` by
19:00 (state, findings, what is left, how to resume).

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (read first)
2. `/home/lstasytis/finn/claude-tools/fifo-sizing/CONTEXT.md`, `HANDOFF.md` sections 1-5
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/issues/002-tfc-throughput/REPORT.md`
   (a similar investigation: per-frame HLS overhead of MVAU_hls/LabelSelect_hls, now fixed; its
   scripts in `issues/002-tfc-throughput/scripts/` simulate single nodes / multiple frames / deep
   FIFOs on an existing stitched IP)

## Symptom
With the committed foldings, cnv FIFO tests pass (rtlsim/estimate 1.001-1.008). With the folding
optimizer's foldings, which are much more balanced (several nodes within a few % of the max), the
stitched-IP rtlsim interval is longer than the estimate; output always correct:

| build | est. cycles (max node) | rtlsim interval | ratio |
|---|---|---|---|
| maximize cnv-w2a2 AUP-ZU3_8GB (fix/005 code) | 27655 (MVAU_2, MVAU_4; then 25607, 25095, 24501, 24307) | 28226 | 0.9798 |
| maximize cnv-w1a2 KV260_SOM (before fixes 002/005) | 12544 (MVAU_hls_1; SWG_rtl_6 12032, MVAU_hls_0 10800, SWG_rtl_3 10616, DWC_5 10368) | 13262 | 0.946 |

Builds: `/home/lstasytis/finn/build/finn_build/fx/fix005/fold-max_bnn-pynq_cnv-w2a2_AUP-ZU3_8GB_*/`,
`/home/lstasytis/finn/build/finn_build/fx/folding/fold-max_bnn-pynq_cnv-w1a2_KV260_SOM_g4trhmom/`
(read-only; copy what you modify). FIFOs sized by `heuristic_analytical`, `rtlsim_batch_size=2`.

## Questions
1. Is it FIFO sizing? Diagnostic only: re-simulate the existing stitched IP with all FIFOs much
   deeper (do NOT use the `largefifo_rtlsim` strategy). If the interval drops to the estimate,
   the analytical sizing under-sizes balanced pipelines: find the edge(s) and the rule, fix on
   `feature/dev/analytical-fifo-sizing`.
2. If not: which node's real interval exceeds `get_exp_cycles()` (SWG_rtl? MVAU_hls with large
   SIMD? DWC? Thresholding?) and by how much; is it per-frame or per-row overhead? Propose/implement
   the minimal estimate fix on the branch that owns it (dev's estimates -> the standalone
   `feature/dev/hls-exp-cycles-overhead`; DWC estimates -> the DWC branch).
3. Measurement artefact (2 frames)? Check 4-8 frames.

## Where to work
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix006 -b fix/006 test/dwc
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix006
cd build/wt/fix006 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
```
`test/dwc` contains all current fixes (folding, DWC, FIFO, HLS overhead). Full re-test (~20-30 min):
`python -m pytest -p no:cacheprovider -v "tests/benchmark/bnn-pynq/test_build_bnn-pynq.py::test_bnnpynq_folding[AUP-ZU3_8GB-cnv-w2a2-maximize]"`
(copy `/home/lstasytis/finn/build/finn_build/fx/dwc/est_bnn-pynq_cnv-w2a2_AUP-ZU3_8GB_*` into your
`$FINN_BUILD_DIR` to reuse the cached checkpoint).

## Pass criterion
Root cause with numbers; a fix after which the two builds above reach rtlsim >= 0.98 x estimate
(re-run at least the cnv-w2a2 maximize test), with the committed-folding FIFO tests still passing
(e.g. `test_bnnpynq_fifo_sizing[AUP-ZU3_8GB-cnv-w1a1-heuristic_analytical]`). If the right fix is a
larger change, report the analysis and options instead.

## Constraints
* Minimal change, match the style, short comments. Never modify dev's RTL `vpc.sv`. Never use `largefifo_rtlsim`. No pushes.
* Self-contained commits per owning branch, say which commit goes where.
* Lint (black 23.3.0 line 100, isort, flake8). Resources: at most 16 threads; MemAvailable > 200 GB;
  kill only your own PIDs (never `pkill -f`); don't touch other worktrees or build dirs (reading is fine).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
