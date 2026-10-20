# Issue 013: node latency in the analytical FIFO sizer (user approved)

Read `issues/010-transformer-maximize-throughput/REPORT.md` ("Maximize: what is left", the two patches in its
`scripts/`). The analytical sizer (`feature/dev/analytical-fifo-sizing`, chained TAVs) has no per-node latency:
InnerShuffle page fill (~263 cycles), dynamic MVAU matrix load, MVAU feature-map blocks, 10-70 cycles pipeline
latency per node. At fast foldings (transformer_benchmark maximize, 1280 cycles/frame) residual shortcuts end up
~300 tokens short (needs 1501-1550, sizer gives 1232) -> rtlsim 0.73x the estimate.

User guidance: **over-estimating a depth is better than under-estimating**, within reason: the total FIFO size must
stay below 1.5x of the optimum (tested later; for now an over-estimate is fine). So prefer a simple, conservative
latency model (e.g. per-op first-output latency / block hooks, or a margin on reconvergent shortcuts derived from
the branch's latency) over an exact one.

Do: implement it on a branch `fix/013` off `feature/dev/analytical-fifo-sizing` (unit tests, docs in
`fifo_sizing_heuristics.rst`); verify transformer_benchmark maximize (`fx/fix010` builds / issue 010's
`scripts/fullsim.sh`) and committed folding still pass; report the total FIFO size change on the models of
`fx/fifo2` / `fx/fifo` feature builds (offline re-sizing is fine) — total should not grow beyond ~1.5x of today's
where today's already passes. Say which commits go on which branch.

## Rules
* Read `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` first (env, worktrees, branches, lint).
* At most 16 threads; start nothing while MemAvailable < 200 GB (an OOM crashes the container); kill only
  recorded PIDs, never `pkill -f`/`pgrep -f` or a `ps | grep` pattern; never use the `largefifo_rtlsim`
  strategy; no pushes; do not touch other worktrees, build dirs or the campaign processes (reading is fine);
  disk ~97 % full, list deletions in REPORT.md. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
* Write REPORT.md next to this file (if refused, put its text into your final answer). Final answer <= 10 lines.
