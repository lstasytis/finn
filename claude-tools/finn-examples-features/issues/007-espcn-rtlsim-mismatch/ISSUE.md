# Issue 007: espcn (KV260_SOM) — stitched-IP rtlsim output differs from the golden output

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this
file when done (if writing the file is refused, put its full text into your final answer).

**Session 2 (2026-10-20): resume.** A first agent worked on this on 2026-10-19 and stopped at the
reboot; its findings and "How to resume" are in `REPORT.md` next to this file (read it right after
the reading list; don't redo what it ruled out). Its worktree `build/wt/fix007` and stitched projects
`/home/lstasytis/finn/build/finn_build/fx/fix007/` survived. Its session scratchpad (incl.
`ctx_hw.npz`) is under `/home/lstasytis/finn/build/home_backup/scratchpad/` (or regenerate with
`pyexec.py`). No deadline, but budget ~8 h of wall clock; long rtlsims run in the background
(`setsid nohup`, record PIDs, the process writes its own PID file). Append a "Session 2" section to
`REPORT.md` (keep session 1's text) and update the Q3 conclusion.
The feature-test campaign runs alongside (large models, up to ~160 threads, ~600 GB reserved):
**at most 16 threads, and start no job while MemAvailable < 200 GB**; disk is 96 % full, keep
your build output small and delete your own scratch projects when done.

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (read first)
2. `/home/lstasytis/finn/claude-tools/fifo-sizing/CONTEXT.md`, `HANDOFF.md` sections 1-5
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/PLAN_per_model_tests.md` sections 2-3

## Symptom
`test_espcn_fifo_sizing[KV260_SOM-heuristic_analytical]` (committed folding JSON, FIFOs sized by
`heuristic_analytical`, test branch `test/fifo`): stitched-IP rtlsim verification FAILS:
`verification_output/verify_stitched_ip_rtlsim_0_FAIL.npy` in
`/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_espcn_KV260_SOM_aiguk3l6/`.
Output (1,3,256,256): 89.6 % of the values equal the golden `tests/benchmark/verification_io/espcn-bsd300_output.npy`
exactly; 10.4 % differ (max |diff| 1.0, i.e. full scale; same fraction in all 3 channels). The errors
are structured: no errors in rows 0-63 (except row 12), from row ~76 on ~15 % of the values per
32-row block, concentrated in the leftmost columns (0-9) with sparse others. The job ran ~3 h
(stitched-IP rtlsim of a 256x256 super-resolution net is slow).
espcn is new from upstream PR #1725 (`tests/benchmark/espcn/`: custom steps lower the
ConvTranspose to FMPadding_Pixel + Im2Col + MatMul). Folding/DWC tests of espcn are still running
(`fold-target__espcn__KV260_SOM` etc. in the campaign logs) and will show whether optimizer
foldings fail the same way.

## Questions
1. Is the golden pair consistent with FINN's own (non-rtlsim) execution? Run the converted /
   folded model in Python/cppsim on the golden input (e.g. the `step_*` checkpoints in
   `intermediate_models/` of the build above, or of the reference estimate build
   `/home/lstasytis/finn/build/finn_build/fx/fifo/est_espcn_KV260_SOM_*`: streamlined and convert
   checkpoints) and compare to the golden output. If Python execution already differs the same way,
   the issue is the model/golden/flow (upstream #1725), not our branches.
2. If Python/cppsim matches the golden but stitched-IP rtlsim does not: which node first diverges?
   Node-by-node rtlsim (`node_by_node_rtlsim` verification) or rtlsim of sub-graphs. Candidates:
   FMPadding_Pixel, ConvolutionInputGenerator (SWG), FIFOs (does the result change with deeper FIFOs?
   diagnostic only, never the `largefifo_rtlsim` strategy), DWCs. The FIFO depths should never change
   the values (only throughput) unless a node misbehaves under backpressure — check that explicitly.
3. Where does the fix belong (dev / #1725 model flow / our FIFO, folding or DWC branches)?

## Where to work
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix007 -b fix/007 test/fifo
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix007
cd build/wt/fix007 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
```
Reuse the build above read-only (copy what you modify). A full rerun of the test takes ~3 h, so
diagnose on the existing artifacts and on small sub-graphs; a full rerun is optional.

## Pass criterion
Root cause with evidence; a fix with stitched-IP rtlsim (of the affected sub-graph or the full
model) matching the golden output, or a clear statement that the golden/flow is wrong upstream and
what the correct reference is.

## Constraints
* Minimal change, match the style, short comments. Never modify dev's RTL `vpc.sv`. Never use the `largefifo_rtlsim` strategy. No pushes.
* Self-contained commits per owning branch; say which commit goes where.
* Lint (black 23.3.0 line 100, isort, flake8). Resources: at most 16 threads; MemAvailable > 200 GB;
  kill only your own PIDs (never `pkill -f`); don't touch other worktrees or build dirs (reading is fine).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
