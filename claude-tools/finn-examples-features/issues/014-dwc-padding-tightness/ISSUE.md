# Issue 014: DWC padding is rarely chosen — make the optimizer use it to tighten over-throughput nodes (user)

Context: `feature/dev/set-folding-optimizer` (optimizer, `folding_maximum_padding`) + `feature/dev/generalized-datawidthconverter`
(padding/cropping DWCs). In today's feature tests `test_<m>_dwc` skips on almost every model: the padded search at
the reference target inserts no padding DWC (seeds 1-8 tried). The user's past tests showed padding is almost
always a win, so suspect the optimizer does not exploit it: with a global target set by the bottleneck, nodes whose
unpadded factors overshoot the target (run faster than needed because no divisor lands closer) should use padding to
reach the smallest PE*SIMD that still meets the target. Check how the padded search, the cost model, the
"padded seeded by unpadded, kept only if no worse" rule (`_better`), repair/polish and `_fold_at_best` interact,
and fix what prevents it (e.g. candidate sets, comparison rules, the cost of padding DWCs/FIFOs).

New acceptance rule (user): a folding slightly below the target counts as a win if its resource reduction (%) vs the
hand-tuned/committed folding is larger than its throughput loss (%), as long as the throughput stays within 10 % of
the target. Implement it where it belongs (optimizer comparison and/or the per-model DWC test assertions in
`tests/benchmark/<model>/test_build_<model>.py`, integration branch `dev-finnexamples-features`; keep tests
self-contained per model file) and report per model: unpadded vs padded folding at the reference target
(cycles, LUT/BRAM/URAM/DSP, number of padded nodes) — estimate-only (`folding` step + estimate reports) is enough
for the comparison, rtlsim only for one or two models.

Work on branches `fix/014-folding` (off `feature/dev/set-folding-optimizer`) / `fix/014-dwc` (off the DWC branch) as
needed; worktree e.g. `build/wt/fix014` from `test2/dwc` for the evaluation.

## Rules
* Read `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` first (env, worktrees, branches, lint).
* At most 16 threads; start nothing while MemAvailable < 200 GB (an OOM crashes the container); kill only
  recorded PIDs, never `pkill -f`/`pgrep -f` or a `ps | grep` pattern; never use the `largefifo_rtlsim`
  strategy; no pushes; do not touch other worktrees, build dirs or the campaign processes (reading is fine);
  disk ~97 % full, list deletions in REPORT.md. Commit messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
* Write REPORT.md next to this file (if refused, put its text into your final answer). Final answer <= 10 lines.
