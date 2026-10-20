# Issue 012: two dev fixes as separate, documented PR branches (user approved)

1. **MVU output queue overflow (issue 007).** Read `issues/007-espcn-rtlsim-mismatch/ANALYSIS.md` §4-7 and
   REPORT.md "Session 3". Implement the two-line fix in `finn-rtllib/mvu/mvu_vvu_axi.sv` (latency formula for the
   deeper non-power-of-2-SIMD core of upstream e1561a40f; `MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1` for the registered
   OLock) on a new branch `feature/dev/mvu-output-queue` off `upstream/dev`. Add the reproducer as a FINN test
   (e.g. `tests/fpgadataflow/test_fpgadataflow_mvau.py`: SF = 1 MVAU_rtl, stitched-IP rtlsim under output
   backpressure, fails without the fix), resolve the open DSP58 value check, and verify espcn end to end: the failing
   build's design (committed folding, analytic FIFOs) with the fixed RTL must match the golden output (use the
   issue-007 tooling, or the feature test `test_espcn_fifo_sizing[KV260_SOM-heuristic_analytical-small]` /
   `-full` on a worktree that merges the fix into `test2/fifo`).
2. **StreamingSplit/Concat at 1 word/cycle (issue 010).** The uncommitted change in `build/wt/fix010-df` (branch
   `fix/010-dev-dataflow`): `#pragma HLS dataflow disable_start_propagation` in the generated top, as
   DuplicateStreams_hls has. Make it a clean branch `feature/dev/split-concat-dataflow` off `upstream/dev` with
   `get_exp_cycles` = 1 word/cycle and the rtlsim tests (issue 010's `test_fpgadataflow_{split,concat}_exp_cycles_rtlsim`
   with the 1x expectation). Note: `feature/dev/hls-exp-cycles-overhead` has the x5 estimate (correct without the
   pragma); say in the PR text how the two relate.
3. For each branch write a PR description file `issues/012-dev-rtl-hls-fixes/PR_<branch-name>.md` (problem, root
   cause with evidence, the change, tests, risk) — these become separate upstream PRs. Lint, single clean commit
   per branch (plus a test commit if cleaner).

## Rules
* Read `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` first (env, worktrees, branches, lint).
* At most 16 threads; start nothing while MemAvailable < 200 GB (an OOM crashes the container); kill only
  recorded PIDs, never `pkill -f`/`pgrep -f` or a `ps | grep` pattern; never use the `largefifo_rtlsim`
  strategy; no pushes; do not touch other worktrees, build dirs or the campaign processes (reading is fine);
  disk ~97 % full, list deletions in REPORT.md. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
* Write REPORT.md next to this file (if refused, put its text into your final answer). Final answer <= 10 lines.
