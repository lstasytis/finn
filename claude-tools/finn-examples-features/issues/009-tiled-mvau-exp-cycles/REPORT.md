# Report 009: tiled MVAU (TH > 1) expected cycles — fixed
(Written by the orchestrator from the fix agent's hand-back; the agent could not write report files.
Full measurement table: `rtlsim_vs_estimate.md`; scripts: `scripts/`.)

## Findings
1. **dev's `* TH` factor is wrong.** The tiled MVU (`finn-rtllib/mvu_tiled`) retires one PE x SIMD tile
   per cycle and reuses it for TH input vectors: a frame costs `NF * SF * numInputVectors` cycles like
   the untiled MVAU; the weight stream is TH times narrower, the compute is not TH times slower.
   Upstream `c15fb8a9a` (drop `* TH`, add the `external_mem` weight-fetch bound) is right as far as it
   goes: bert's four tiled MVAUs measure exactly 49152 cycles/frame (TH 4 and 8); the `external_mem`
   bound matches rtlsim within 1 cycle per fetch.
2. **Two per-group stalls of the tiled MVU that `c15fb8a9a` misses**, from the power-of-two circular
   buffers of its `input_gen` instances, once per group of TH vectors:
   * *activation replay*: the group's `SF*TH` input words are freed only after the last of the NF
     replays, so only `BUF_in - SF*TH` words of the next group are prefetched
     (`BUF_in = 2^ceil(log2(SF*TH+2))`). Stall `max(0, (TH-1)(SF-1) + 3 - (BUF_in - SF*TH))`. Up to
     60 % of a frame (MW 28 / SIMD 4 / TH 4 / NF 1: 90 vs 56). Identical to the term the FIFO branch's
     tree model already had (`mvau_tiled_params`, fitted), now derived from the RTL.
   * *output reorder*: the reorder `input_gen` holds all `NF*TH` words of a group
     (`BUF_out = 2^ceil(log2((NF-1)*TH+3))`) until the last fold arrives, then drains
     `R = NF*TH-(NF-1)` words while the next group produces outputs; if they don't fit the spare slots
     the pipeline freezes: `out = max(0, R - t_block, -spare)`, `spare = BUF_out - NF*TH`,
     `w = max(0, spare + TH - 4)`, `t_block = SF*TH*(w//TH + 1) + w%TH`. Up to 83 % of a frame (SF 1,
     NF 32, TH 8), 13 % for a bert-like 3D shape (MW 32 / MH 128 / SIMD 8 / PE 8 / TH 4 / [1,12,16]:
     13872 vs 12288). Not in any model before.
   * They overlap; per group the larger one counts (SF 7 / NF 16 / TH 4: 17 and 21 -> 21 measured).
   A cycle model of both buffers (`scripts/tiledsim.py`) reproduces every measurement exactly; the
   closed form is exact for power-of-two TH and NF >= 2, within ~10 % of a group for other TH.
3. **bert**: its tiled MVAUs have no stall; estimate `max_cycles` becomes 98304 (`MVAU_rtl_3`,
   dynamic) = stitched-IP rtlsim interval 98304 (`fx/fix001/fifo-heuristic_analytical_bert_VCK190_3wk9udu5`,
   ratio 1.000; was 393216, ratio 4.0). bert has no FINNLoop.
4. `dynamic` (TH 1, 2) and `internal_decoupled` TH=1 unchanged and exact (measured at SIMD/PE 8);
   HLS untouched (TH=1 only); `external_mem` +1 cycle per weight fetch on top of upstream's bound.

## rtlsim vs estimate (excerpt; 115 foldings, single-MVAU stitched-IP rtlsim on xcvc1902)
| mem_mode | MW | MH | SIMD | PE | TH | n_vec | dev est | new est | rtlsim |
|---|---|---|---|---|---|---|---|---|---|
| internal_decoupled | 32 | 32 | 8 | 8 | 2/4/8 | [1,16] | 512/1024/2048 | 256 | 256 |
| internal_decoupled | 384 | 384 | 24 | 16 | 8 | [1,128] (bert) | 393216 | 49152 | 49152 |
| internal_decoupled | 384 | 1536 | 48 | 32 | 8 | [1,128] (bert) | 393216 | 49152 | 49152 |
| internal_decoupled | 28 | 8 | 4 | 8 | 4 | [1,8] replay | 224 | 90 | 90 |
| internal_decoupled | 32 | 128 | 8 | 8 | 4 | [1,12,16] reorder | 49152 | 13872 | 13872 |
| internal_decoupled | 28 | 128 | 4 | 8 | 4 | [1,16] both | 7168 | 1876 | 1876 |
| internal_decoupled | 48 | 6 | 12 | 1 | 12 | [1,24] TH not 2^k | 6912 | 616 | 646 |
| dynamic | 32 | 128 | 8 | 8 | 1 | [1,12,16] (bert) | 12288 | 12288 | 12288 |
| external_mem | 64 | 64 | 16 | 8 | 2 | [1,16] AXI bound | 1024 | 1024 | 1032 |
All 93 power-of-two-TH foldings within 1 % (most exact), the 22 others within 7 %.

## Fix
* **`feature/dev/mvau-exp-cycles`** = `upstream/dev` (`11ebf1abb`) + `187efdac7` (cherry-pick of
  upstream `c15fb8a9a`, unchanged) + `cbd53df4c` "[MVAU] Tiled MVAU_rtl: model the group stalls in
  get_exp_cycles" (`tiled_mvau_group_stall(sf, nf, th)`, `MVAU.get_tiled_group_stall_cycles()`, added
  as `(n_vec // TH) * stall` in `get_exp_compute_cycles()`; tests `test_mvau_rtl_exp_cycles` (21
  measured foldings, no Vivado) and `test_fpgadataflow_mvau_tiled_exp_cycles_rtlsim` (3 foldings)).
  Both commits belong on dev; our branches get them by merge. Worktree `build/wt/fix009`.
* **`fix/009-fifo`** = `feature/dev/analytical-fifo-sizing` (`76506974b`) + the two commits
  (`702496807`, `9350f8fbf`) + `e8c2cc992` "FIFO sizing: tiled MVAU tree model uses the shared group
  stall" (`mvau_tiled_params` takes the replay stall from `tiled_mvau_group_stall`; `mvau_tiled_tree`
  declines to rtlsim where the reorder stall dominates; test asserts the declines). Worktree
  `build/wt/fix009-fifo`.

## FIFO sizing
The tiled tree model was right in its regime (all 10 `TILED_TAV_CONFIGS` pass against node rtlsim with
the shared stall function); its gap was the reorder-stall regime (SF small, NF*TH just under a power of
two), now declined to rtlsim characterization. `derive_token_access_vectors` is TH-agnostic and right;
`dynamic_in1_tav` is independent of TH.

## FINNLoop (proposal only, not applied)
ZU3 MLO build (`test_finnloop_end2end_mlo_ddr` config without bitfile, `fx/fix009/i009_mlo_xsm4l9nv`):
body critical path 342, max_cycles 108; FINNLoop rtlsim 779 cycles for one input; new upstream estimate
1106 (upper bound, 1.42x), old 678 (0.87x, under-estimate). For a bert-sized body `critical_path_cycles`
(sum of node cycles) is 6.7x the measured latency and 10.7x the interval. Proposal: body latency =
`max_cycles + fill` (sum of first-output latencies), keep `latency * iteration + 40 * (iteration - 1)`,
and document the value as a single-input latency, not a throughput interval.

## Tests
`fix009`: `test_mvau_rtl_exp_cycles` 21 passed, tiled rtlsim test 3 passed (256/300/98 = estimate),
`test_set_folding.py` passed. `fix009-fifo`: the same plus
`test_fpgadataflow_analytical_characterization_mvau_tiled` 10/10, tree-model decline/token tests. Lint
clean. Scratch builds deleted (`fx/fix009` keeps the 8 MB MLO build).
