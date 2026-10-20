# Issue 016: Split/Concat FIFO tree models vs the dataflow pragma

Read first: `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md`, then
`issues/012-dev-rtl-hls-fixes/REPORT.md` and `PR_feature-dev-split-concat-dataflow.md`.

`feature/dev/split-concat-dataflow` (bcb838bbb) adds `#pragma HLS dataflow disable_start_propagation` to the
StreamingSplit_hls / StreamingConcat_hls tops: Interval 5 -> 1, latency 4 -> 2. The FIFO branch
(`feature/dev/analytical-fifo-sizing`) has `get_tree_model()` for both (`src/finn/custom_op/fpgadataflow/split.py`,
`concat.py`) hard-coded to `ii = 5` and the cycle offsets of the unpipelined top. The integration branch
`dev-finnexamples-features` (worktree `build/wt/integ`) and `test2/*` now contain both, so the tree models are wrong
there. No tree-vs-rtlsim characterization test exists for Split/Concat.

Tasks:
1. Measure the real schedule (node-level rtlsim TAVs, as `derive_token_access_vectors` does) of both ops with and
   without the pragma for a few configs (equal/unequal ChannelsPerStream, SIMD 1/2/4, numInputVectors), and compare
   with the tree models.
2. Make the tree models correct for both cases without coupling the two PRs badly: e.g. the tree model takes the
   interval/offsets from a small method on the HLS op (default: the unpipelined values, so the FIFO PR stays correct
   on plain dev) that the pragma version overrides — decide what is cleanest and say which commit goes on which
   branch (FIFO PR, pragma PR, or an integration-only adaptation). Over-estimating FIFO depth is acceptable (user).
3. Add tree-vs-rtlsim characterization tests for Split/Concat (like the other ops' `test_fpgadataflow_analytical_characterization_*`).
4. Land on fix branches (`fix/016-fifo` off the FIFO branch, `fix/016-pragma` off the pragma branch if needed) and
   verify on the integration branch (unit tests; transformer_benchmark FIFO/fold-target feature test small if cheap).

Rules: at most 16 threads; start nothing while MemAvailable < 200 GB; kill only recorded PIDs (never `pkill -f`/`pgrep
-f` or a `ps | grep` pattern); no RTL/HLS library changes beyond the pragma already on its branch; never
`largefifo_rtlsim`; no pushes; don't touch other worktrees, build dirs or the campaign; disk ~97 % full, list
deletions. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Write REPORT.md next to
this file (if refused, put its full text into your final answer after a <= 10-line summary).
