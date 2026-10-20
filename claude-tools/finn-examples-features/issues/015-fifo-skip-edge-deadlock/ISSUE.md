# Issue 015: analytic FIFO sizing deadlocks transformer residual skips (radioml, vision)

Read first: `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md`, then
`issues/011-transformer-radioml-wrong-output/REPORT.md` (section 2 + scripts `fifo_override.py`, `tav_diag.py`) and
`issues/013-fifo-latency-model/REPORT.md` (the node-latency change now on `feature/dev/analytical-fifo-sizing`).

Problem: `heuristic_analytical` sizes the MLP residual skip FIFO below one token: radioml committed folding FIFO_90
(Thr_13 -> ElementwiseAdd_hls_4) = 20 words while the main branch (DWC 8->384 bit needs all 48 words of a token,
MVAU_rtl_11 -> DWC_26 needs 64 -> MVAU_rtl_12) must read a whole token before emitting -> the fork blocks -> deadlock.
vision fold-target: skip FIFOs 92 / 108, same. MVAU tree-model TAVs put the first output 44-953 cycles before the
first input; the fork is treated as transparent.

Tasks:
1. Check whether the issue-013 latency change (now on feature/dev/analytical-fifo-sizing and `dev-finnexamples-features`,
   worktree `build/wt/integ`) already sizes these edges enough (offline re-sizing or the feature test
   `test_transformer_fifo_sizing[ZCU104-transformer_radioml-heuristic_analytical-small]`).
2. If not, fix it on a branch `fix/015` off `feature/dev/analytical-fifo-sizing`: user guidance — over-estimating is
   better than under-estimating (total FIFO size within 1.5x of optimum later), so a conservative floor is fine, e.g.
   on fork -> join skip edges at least the tokens the other branch must read before its first output (DWC input
   ratio, MVAU input window) plus margin, and/or causal MVAU profiles. Unit test + docs (`fifo_sizing_heuristics.rst`).
   Report the total FIFO size change on the models (offline, like issue 013's `compare_all.sh`).
3. Verify: radioml and vision FIFO / fold-target feature tests (small first) no longer hang and produce the golden
   output (the GAP + Requant fixes of issue 011 are on the integration and test2 branches).
4. Secondary, if time allows: (a) radioml fold-target/max reach only 0.2x of the estimate in rtlsim (3255 vs 16276 fps)
   on a build that predates the issue-010 fixes (Split/Concat estimate, FIFO phase lead) — re-check on the current
   integration branch and find the bottleneck node if it persists; (b) `heuristic_rtlsim` fails in per-node
   characterization on radioml (a node times out, probably the two-input ElementwiseAdd_hls whose timing harness feeds
   only one input).

Rules: at most 16 threads; start nothing while MemAvailable < 200 GB; kill only recorded PIDs (never `pkill -f`/`pgrep
-f` or a `ps | grep` pattern); no RTL/HLS library changes; never `largefifo_rtlsim`; no pushes; don't touch other
worktrees, build dirs or the campaign; disk ~97 % full, list deletions. Commit messages end with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Write REPORT.md next to this file (if refused, put its full
text into your final answer, after a <= 10-line summary).
