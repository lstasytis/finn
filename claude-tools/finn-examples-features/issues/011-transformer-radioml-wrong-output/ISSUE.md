# Issue 011: transformer_radioml — stitched-IP rtlsim output wrong (and a hang with analytic FIFOs)

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this file
when done (if writing it is refused, put its full text into your final answer). Budget ~6 h wall clock.

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, rules; read first;
   current test branches: `test2/{fifo,folding,dwc}` in `build/wt/{fifo2,folding2,dwc2}` on the new
   upstream base; integration branch `dev-finnexamples-features` in `build/wt/integ`)
2. `tests/benchmark/transformer/test_build_transformer.py` and `custom_steps_transformer.py` (new in
   upstream PR #1725, branch `upstream/update/finn-examples`; the per-model feature tests at the end of
   the test file are ours), `tests/benchmark/benchmark_helpers.py` (`run_feature_build` pieces:
   `cached_build`, `resume_checkpoint`, `feature_steps`, `feature_metrics`).

## Symptom (ZCU104, model `transformer_radioml`, golden I/O `tests/benchmark/verification_io/transformer_radioml_{input,output}.npy`)
* `test_transformer_folding[ZCU104-transformer_radioml-target]` and `[...-maximize]` (optimizer foldings,
  branch `test2/folding`): stitched-IP rtlsim output **identical in both builds** and wrong: golden
  `[-11.63 ... 3.54 5.99 ... ]` (argmax 11, looks like saturated logits), rtlsim
  `[-0.82 -0.27 ... 0. 0.27 ...]` (argmax 15, max |diff| 11.9) -> deterministic, not timing.
  Builds: `/home/lstasytis/finn/build/finn_build/fx/folding2/fold-target_transformer_transformer_radioml_ZCU104_eoq3qgo5/`,
  `.../fold-max_transformer_transformer_radioml_ZCU104_82tmyeqv/`.
* `test_transformer_fifo_sizing[ZCU104-transformer_radioml-heuristic_analytical]` (committed folding,
  branch `test2/fifo`): stitched-IP rtlsim **times out with no output** (`m_axis_0` watchdog after
  182795 cycles). Build under `/home/lstasytis/finn/build/finn_build/fx/fifo2/fifo-heuristic_analytical_transformer_transformer_radioml_ZCU104_*`.
* transformer_benchmark (same file, 8-bit, no `requant_bitwidth_threshold` override) produces correct
  output (its only problem is throughput, issue 010). radioml/vision/language use
  `requant_bitwidth_threshold=8`.

## Questions
1. Baseline: does the model pass with **pure upstream**? Build upstream's own flow on
   `upstream/update/finn-examples` (a fresh worktree; `configure_build` of the upstream test file,
   committed folding, upstream's default FIFO sizing, verify steps up to `stitched_ip_rtlsim`, no
   bitfile: `generate_outputs` = estimate reports + stitched IP). Also check the Python/cppsim verify
   steps (`streamlined_python`, `folded_hls_cppsim`, `node_by_node_rtlsim`) to see where the output
   first diverges from the golden.
2. If upstream passes: which of our changes breaks it (bisect over the merges on
   `dev-finnexamples-features`: FIFO branch, folding branch, DWC branch, HLS-overhead, MVAU exp cycles;
   or the feature-test path itself: resuming from the `phase_convert_to_hardware` checkpoint of the
   cached estimate build). If upstream fails too: where (which step/node), and is it the model export,
   the golden pair, or a FINN bug (e.g. Requant, which the 8-bit outputs use).
3. The hang with analytic FIFOs (committed folding): deadlock from undersized FIFOs (which edge?), or the
   same functional problem (a node that never emits)? Check node-level behaviour and the FIFO depths
   against what `heuristic_rtlsim`/upstream's sizing gives (diagnostic only: never use the
   `largefifo_rtlsim` strategy).
4. Fix what belongs to us on the owning branch (say which), or document an upstream bug with a minimal
   reproducer.

## Where to work
```bash
cd /home/lstasytis/finn
git worktree add build/wt/fix011 -b fix/011 dev-finnexamples-features   # and e.g. build/wt/fix011-up on upstream/update/finn-examples
claude-tools/finn-examples-features/tools/wtsetup.sh build/wt/fix011
cd build/wt/fix011 && source /home/lstasytis/finn/claude-tools/finn-examples-features/tools/wtenv.sh .
export FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix011
```

## Pass criterion
Root cause with evidence (the first diverging step/node), and a fix with stitched-IP rtlsim matching the
golden output, or a clear statement that it is upstream's, with a reproducer.

## Constraints
* Minimal change, match the style, short comments. Never modify dev's RTL/HLS sources. Never use the
  `largefifo_rtlsim` strategy. No pushes.
* Resources: at most 16 threads; start nothing while MemAvailable < 200 GB (an OOM crashes the
  container); kill only your own recorded PID trees, never `pkill -f`/`pgrep -f`; a feature-test campaign
  and other agents run alongside: don't touch other worktrees, build dirs or processes (reading is
  fine). Disk is ~97 % full: keep outputs small, delete your own scratch projects (list deletions in
  REPORT.md).
* Lint (black 23.3.0 line 100, isort, flake8; `.lintenv`, see GENERAL.md).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; say in REPORT.md
  which commit belongs on which branch.
