# Issue 010: transformer_benchmark maximize folding — stitched-IP rtlsim 0.2x the estimate

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this file
when done (if writing it is refused, put its full text into your final answer). Budget ~6 h wall clock.

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, rules; read first;
   note: the current test branches are `test2/{fifo,folding,dwc}` in `build/wt/{fifo2,folding2,dwc2}`,
   on the new upstream base, integration branch `dev-finnexamples-features`)
2. `issues/002-tfc-throughput/REPORT.md` (per-frame HLS overhead, how node cycles were checked against rtlsim),
   `issues/006-balanced-folding-throughput/REPORT.md` (balanced foldings, FIFO-related losses),
   `issues/009-tiled-mvau-exp-cycles/REPORT.md` (MVAU exp cycles vs rtlsim, method and scripts),
   `issues/007-espcn-rtlsim-mismatch/REPORT.md` (OuterShuffle_hls reaches only II=3, 3x its estimate)
   — all under `/home/lstasytis/finn/claude-tools/finn-examples-features/`.

## Symptom
New finn+ transformer models (upstream PR #1725, `tests/benchmark/transformer/`, ZCU104, 10 ns).
`test_transformer_folding[ZCU104-transformer_benchmark-maximize]` (optimizer maximize mode, FIFOs
`heuristic_analytical`, branch `test2/folding`): output correct, but stitched-IP rtlsim 9765.6 fps
(10240 cycles/frame) vs the build's estimate 48828 fps (2048 cycles/frame) = 0.2x. Dozens of nodes are
at exactly 2048 estimated cycles (Thresholding_rtl, DuplicateStreams_hls, DWCs, MVAU_rtl incl. the
dynamic attention MVAUs, StreamingSplit_hls, ...). Build:
`/home/lstasytis/finn/build/finn_build/fx/folding2/fold-max_transformer_transformer_benchmark_ZCU104_j33hxjs7/`
(intermediate models, reports, `report/feature_metrics.json`).
Related, smaller: with the committed folding (and with the optimizer at the committed target) the same
model reaches 0.965x (interval 67895 vs 65536; bottleneck 4x MVAU_rtl MW=MH=32 SIMD=PE=1 n_vec [1,64]
internal_decoupled) — the issue-009 agent was asked to measure that MVAU config; check its REPORT.md
before duplicating.

## Questions
1. Which node(s) really bound the throughput at this folding? Measure per-node steady-state cycles
   (single-node or small sub-graph stitched-IP rtlsim with several frames, like issue 009's
   `scripts/tiled_sweep.py`) for every op type at its folded configuration, vs `get_exp_cycles()`.
   Candidates: InnerShuffle/OuterShuffle (after `step_transpose_decomposition`; OuterShuffle II=3 is
   known), dynamic MVAU (`dynamic_load` refill: issue 001 found it cannot refill when
   `numInputVectors[-1] < SIMD`), HWSoftmax, LayerNorm, StreamingSplit/Concat, DuplicateStreams,
   ElementwiseBinary, Requant, DWCs, Thresholding.
2. Fix the estimate(s) that are wrong (`get_exp_cycles` of the op; that belongs on dev: a new commit on
   a `feature/dev/<op>-exp-cycles`-style branch off `upstream/dev`, or on
   `feature/dev/hls-exp-cycles-overhead` / `feature/dev/mvau-exp-cycles` if it extends those), so that
   the optimizer sees the real bottleneck. If a node cannot reach its estimate for a hardware reason that
   the folding could avoid (e.g. a SIMD the node cannot sustain), say so: the optimizer may need to
   exclude such values (`feature/dev/set-folding-optimizer`).
3. Rerun the maximize folding (estimate-only: `run_feature_build(..., estimate_only=True)` style or the
   folding step alone) with the fix and, if time allows, the full test; report rtlsim vs estimate.

## Where to work
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix010 -b fix/010 test2/folding
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix010
cd build/wt/fix010 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix010
```

## Pass criterion
Per-node rtlsim vs estimate table for the folded transformer_benchmark; the wrong estimate(s) fixed with
tests; the maximize folding's estimate within ~2-5 % of its stitched-IP rtlsim (or a clear statement of
what is left).

## Constraints
* Minimal change, match the style, short comments. Never modify dev's RTL/HLS sources. Never use the
  `largefifo_rtlsim` strategy. No pushes.
* Resources: at most 16 threads; start nothing while MemAvailable < 200 GB (an OOM crashes the
  container); kill only your own recorded PID trees, never `pkill -f`/`pgrep -f`; a feature-test campaign
  and another agent run alongside: don't touch other worktrees, build dirs or processes (reading is
  fine). Disk is ~97 % full: keep outputs small, delete your own scratch projects (list deletions in
  REPORT.md).
* Lint (black 23.3.0 line 100, isort, flake8; `.lintenv`, see GENERAL.md).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; say in
  REPORT.md which commit belongs on which branch.
