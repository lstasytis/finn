# 024: FIFO sizing PR blockers (yolov8 hang, cnv-w2a2 0.98, resnet50 / transformer wrong output)

Context (2026-10-26): `feature/dev/analytical-fifo-sizing` (tip 6f734bc6e) is about to go up as a PR into
`feature/analytical-fifo-sizing` (= upstream PR Xilinx/finn#1185). The PR argues that the analytical sizer
(`auto_fifo_strategy="heuristic_analytical"`) neither deadlocks nor loses throughput on the finn-examples models.
Campaign results: `claude-tools/finn-examples-features/STATUS.md` (variant 2, "FIFO analytical" / "FIFO rtlsim"
columns; logs `build/finn_build/fx/logs/campaign/v2__fifo*__*.log`, builds in `build/finn_build/fx/fifo2/`).
Read `GENERAL.md` first; FIFO sizing internals: `docs/finn/fifo_sizing_heuristics.rst` on the branch and the
reports of issues 006, 013, 015, 016, 018 (tools for "which rule sets this depth" in `issues/018-*/scripts/`).

## Part A (one agent): yolov8 hang, cnv-w2a2 at 0.9797

1. **quantyolov8_4w4a small (U55C), heuristic_analytical: stitched-IP rtlsim produced no output word in 383 M
   cycles** (`m_axis_0 timeout`; build `fx/fifo2/fifo-heuristic_analytical-small_yolov8_quantyolov8_4w4a_U55C_nvv5p58r`,
   log `v2__fifo__yolov8__U55C-quantyolov8_4w4a__small.log`). Decide deadlock vs. just slow (expected cycles per frame
   from the estimate; first-output latency). If deadlock: find the join/branch whose FIFO is too small (yolov8 has
   C2f Split/Concat and residual adds), fix the sizer on `feature/dev/analytical-fifo-sizing` (smallest rule change,
   unit test next to the existing ones in `tests/fpgadataflow/test_fifosizing_chained_tav.py`), check that the depths
   of the models that already pass do not change (or still pass), and verify yolov8 small with a stitched-IP rtlsim
   run (a sub-graph or a reduced-input reproducer is fine for iteration; the final check is the feature test).
2. **cnv-w2a2 small (AUP-ZU3_8GB and U55C): rtlsim 0.9797x the estimate** with both `heuristic_analytical` and
   `heuristic_rtlsim` (identical ratio). Decide whether FIFO depth costs the 2 % (re-sim with all FIFOs larger) or the
   estimate is optimistic (e.g. an HLS per-frame overhead). Fix the side that is wrong; if it is the estimate (not a
   FIFO matter) document it for the PR and stop.

## Part B (one agent): wrong outputs

3. **resnet50 full (V80), heuristic_analytical: stitched-IP rtlsim output differs from the golden** (small passes;
   log `v2__fifo__resnet50__V80.log`). Known from the islands work: the committed resnet50 folding puts a DWC between a
   depthwise SWG (SIMD 4 / 2048) and Pool_hls (PE 2) without reordering, which computes wrong pool outputs
   (`git show feature/rapidwright-islands:experiments/dynarapid/NOTES_ISLANDS.md`, section "2026-10-12 (evening)",
   `experiments/dynarapid/fix_pool_swg_simd.py` on that branch). Confirm or refute that this is the cause (node-level
   python vs rtlsim of the pool nodes, no full rtlsim needed); it is not a FIFO matter if confirmed. Document for the PR.
4. **transformer_radioml / vision small: output wrong, throughput 1.00x** with both heuristic strategies (issue 019,
   `issues/019-transformer-regression/ISSUE.md`, never finished). Finish 019's bisection; the key question for the PR is
   whether FIFO depths are involved (a node that computes wrong data only under certain backpressure, like issue 007's
   MVU queue, would make depths matter). Fix on the owning branch; if it is a FIFO-branch commit, fix it there.

## Rules

* Machine: the campaign is still running; start a job only while the 1-min load + its threads <= 115 and
  MemAvailable > 200 GB; at most 24 threads per agent. Never `pkill -f`/`pgrep -f`; kill recorded PID trees. Never
  `largefifo_rtlsim`. Never OOM the machine.
* Own scratch worktrees (`build/wt/fix024a*`, `fix024b*`), own `FINN_BUILD_DIR` (`build/finn_build/fx/fix024a|b`).
  Don't touch the campaign, other worktrees or build dirs.
* Code fixes: commit on the owning branch (`feature/dev/analytical-fifo-sizing` for the sizer), then merge it into
  `dev-finnexamples-features` and the `test2/*` branches like the earlier fixes (GENERAL.md: DWC invariant after
  merging). Lint with `.lintenv`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  No pushes.
* Write `REPORT_A.md` / `REPORT_B.md` in this folder; final answer <= 10 lines.
