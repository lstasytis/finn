# Issue 009: tiled MVAU (TH > 1) expected cycles are wrong

You are a fresh agent with no prior context. Work autonomously; write `REPORT.md` next to this file
when done (if writing it is refused, put its full text into your final answer). Budget ~6 h wall clock.

## Reading list
1. `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md` (env, worktrees, rules; read first)
2. `/home/lstasytis/finn/claude-tools/finn-examples-features/issues/001-bert-fifo/REPORT.md` (bert's
   MVAUs, the FIFO tree model / characterization of MVAU)
3. `/home/lstasytis/finn/claude-tools/finn-examples-features/issues/002-tfc-throughput/REPORT.md`
   (how per-node exp cycles were checked against rtlsim before; the +7/+4 HLS overhead branch)

## Symptom
bert (VCK190) stitched-IP rtlsim runs 4.0x faster than FINN's estimate (rtlsim interval 98304 cycles
vs estimate 393216): its tiled `MVAU_rtl` nodes (attribute `TH` = 4) have `get_exp_cycles()` =
`MH/PE * MW/SIMD * prod(numInputVectors) * TH`. The user confirmed the estimate is wrong and wants a
reliable fix, because the folding optimizer (cycle constraint) and the analytical FIFO sizing
(`heuristic_analytical`: tree model / token access vectors, slack rules use exp cycles) both rely on
it for models with tiled MVAUs (bert, the new finn-plus transformers in upstream PR #1725:
`tests/benchmark/transformer/`, models `transformer_{benchmark,radioml,vision,language}`).

Upstream (auphelia) has a candidate fix on `upstream/fix/exp_cycles`, commit `c15fb8a9a`
("[CustomOps] Update MVAU and finnloop exp cycles"): drops the `* TH` factor from the compute cycles,
adds `get_exp_weight_fetch_cycles()` for `mem_mode="external_mem"` (weight fetch can bound the rate),
and changes FINNLoop's estimate to `body_latency * iteration + 40 * (iteration - 1)`.

## Tasks
1. Branch `feature/dev/mvau-exp-cycles` from `upstream/dev` (fetch first:
   `git fetch https://github.com/Xilinx/finn.git '+refs/heads/dev:refs/remotes/upstream/dev' '+refs/heads/fix/exp_cycles:refs/remotes/upstream/fix/exp_cycles'`),
   cherry-pick `c15fb8a9a`; worktree `build/wt/fix009` (`tools/wtsetup.sh`).
2. Measure: single-node rtlsim (stitched-IP rtlsim of a one-node model, or the node-level rtlsim used by
   characterization) of tiled MVAU_rtl for a grid of TH in {1,2,4,8}, PE/SIMD, MW/MH,
   numInputVectors (incl. 3D inputs like bert's, `[1, 128, 384]`-style), mem_modes
   `internal_decoupled`, `dynamic`, `external_mem` where supported, several frames back to back:
   steady-state cycles per frame vs `get_exp_cycles()`. Include bert's actual MVAU configurations
   (from `/home/lstasytis/finn/build/finn_build/fx/fifo/est_bert_VCK190_*/intermediate_models/` or the
   folded model of a bert feature build; note bert's MVAUs sit in a FINNLoop body).
3. Fix `get_exp_cycles` (on top of `c15fb8a9a`, or replace it if it is wrong) so that it matches
   rtlsim within a few % (state the residual; a constant per-frame pipeline overhead is fine to add
   if it is real). Keep `_hls` behaviour unchanged except where TH applies (HLS supports TH=1 only).
4. FIFO sizing: check the MVAU tree model / token access vectors (`get_tree_model`,
   `derive_token_access_vectors`, `dynamic_in1_tav` on `feature/dev/analytical-fifo-sizing`) for
   TH > 1 against rtlsim characterization (period and in/out token curves). If they are wrong for
   TH > 1, fix them in a separate commit on a branch off `feature/dev/analytical-fifo-sizing`
   (`fix/009-fifo`).
5. FINNLoop part of `c15fb8a9a`: check against bert's stitched-IP rtlsim (interval 98304 cycles,
   build `/home/lstasytis/finn/build/finn_build/fx/fix001/fifo-heuristic_analytical_bert_VCK190_3wk9udu5/`)
   whether the new FINNLoop estimate is right for the throughput (it was written for single-input
   latency). Report the numbers; propose, don't silently change upstream's FINNLoop semantics.
6. Unit tests in `tests/fpgadataflow/test_fpgadataflow_mvau.py` for the new expectations (fast,
   no Vivado where possible; one rtlsim-backed test is fine).

## Pass criterion
Per-node rtlsim vs `get_exp_cycles()` table over the grid within a few %; bert's estimate consistent
with its stitched-IP rtlsim interval; FIFO tree model correct for TH > 1 (or fixed); tests; lint.

## Constraints
* Minimal change, match the style, short comments. Never modify dev's RTL (`vpc.sv`, the MVU RTL).
  Never use the `largefifo_rtlsim` strategy. No pushes.
* Resources: at most 16 threads; start nothing while MemAvailable < 200 GB (an OOM crashes the
  container); kill only your own recorded PID trees, never `pkill -f`/`pgrep -f`; a feature-test
  campaign and another agent run alongside: don't touch other worktrees or build dirs (reading is
  fine). Disk is ~96 % full: keep outputs small, delete your scratch projects. Builds go to
  `FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/fix009`.
* Lint (black 23.3.0 line 100, isort, flake8; `.lintenv`, see GENERAL.md).
* Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; say in
  REPORT.md which commit belongs on which branch.
