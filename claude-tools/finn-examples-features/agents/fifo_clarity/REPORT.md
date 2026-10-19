# Report: chained-TAV FIFO sizing made understandable

Branch `feature/dev/analytical-fifo-sizing-clarity` (worktree
`/tmp/finn_dev_lstasytis/worktrees/fifo-clarity`), based on 506ac5832. Nothing pushed.

## Commits

```
f7ca3e9a7 Benchmark feature flows: more models for quick-mode regressions
3ab5b5101 FIFO sizing: split the chained-TAV pass into named steps
8d0248e56 FIFO sizing: drop three rules that change no depth
069cfe26d FIFO sizing: depth regression test for the chained-TAV sizer
a9704fe12 FIFO sizing: document the heuristics with measured examples
```

## What was simplified

`src/finn/transformation/fpgadataflow/derive_characteristic.py`: 802 -> 763 lines, but the
line count understates it; the structure changed from one 270-line function with six
closures to a sequence of top-level, individually documented steps:

| before (55b0b08ce) | after |
|---|---|
| `derive_chained_tav_depths` (266 lines): TAV reading, periods, pacer walk, branch search, schedule propagation and depth arithmetic interleaved; closures `chain_period`, `idle_window_floor`, `absorbed_frame_floor`, `depth_from_peak`, `token_times`, `reachable`; two dicts `arrival`/`supply` with positional tuples | `read_node_tavs` (step 1), `node_periods` (2), `pacer_drivers` (3), `reconvergent_branch_nodes` (4), `graph_input_schedule`, `local_clock`, `token_times`, `frame_tokens` (5), `idle_window_floor`, `absorbed_frame_floor`, `edge_depth` (6); one `Edge(writes, per_frame, t_up, t_prop)` record; the main loop is ~70 lines |
| `pacer_period = max([global_period] + periods) * 0.995` | `global_period * PACER_TOLERANCE` (what the max always computed) |
| QUIRK `input_rows_collapsed` | removed (identical depths on all 10 models) |
| QUIRK `inherited_fork_join_tav_lengths` (47 lines) | removed (identical depths, incl. resnet50) |
| `_TRANSPARENT_OPS` (ElementwiseAdd by op type) | removed: transparent == no TAV (identical depths, incl. resnet50's two unary adds) |
| `CAP_MARGIN`, two passes (causal / non-causal), `idle_window_floor`, join rule | kept: each changes resnet50 depths, which cannot be simulated here (see open questions) |

Every rule is now one named function or one labelled block in `edge_depth`, with the
constants `PACER_TOLERANCE`, `THROTTLED_CAP`, `CAP_MARGIN` documented at the top of Stage 4.

## The heuristics (one line each; details and measured examples in `docs/finn/fifo_sizing_heuristics.rst`)

Core (not a heuristic): per-node local schedule -> wall clock by the max-plus recurrence
`R(c) = max(R(c-1)+1, arrival of tokens read by c)`; depth = peak of written - read, periodic
over the global period. On cnv-w1a1 every non-minimal depth is exactly this peak.

1. **Paced graph input**: the source delivers one frame per global period, evenly spaced.
   cnv-w1a1 DWC_0 -> CIG_0: 60 instead of 638; 12255 fps either way; 11.9 vs 25.5 kbit.
2. **Causal writes** (two passes, max per edge): compacting nodes cannot emit output j before
   input ceil((j+1) N_in/N_out). One token on one edge of cnv/gtsrb/mobilenet; on resnet50
   the non-causal pass is larger on 7 edges (e.g. shortcut 14106 vs 11870) - not simulated.
3. **Two-frame split**: fill frame and steady frame evaluated separately, chain edges take the
   max, join/branch edges the fill frame. cnv-w1a1 without: depths double (52->104, 72->142,
   11.9 -> 18.4 kbit), same 12255 fps.
4. **Pacer tolerance 0.995**: bottleneck several nodes wide. vgg10 five edges 2 -> 3 without;
   mobilenet 9432 -> 64586 total depth without.
5. **Slack relaxation** (+ reset past the pacer): a producer chain that finishes early can be
   blocked; `per_frame (1 - t_up/T)` tokens deducted. gtsrb Pool_0 -> CIG_3 176 -> 59;
   cnv-w2a2 1152 -> 594; mobilenet 64832 -> 9432. gtsrb rtlsim without it: 8134 fps either way, 10.8 vs 9.9 kbit.
6. **Burst floor**: depth >= min(peak, consumer's demand above the average rate line). The
   rule that sizes most storage. cnv-w1a1 without: 60->2, 52->28, 72->66 and throughput
   12255 -> 12077 fps (-1.5 %). Known over-size: kws Thresholding_0 -> MVAU_0 = 428 while
   the reference needs 2 (25471 fps both; 4.3 vs 0.9 kbit) because the testbench source
   saturates whereas the sizer assumes a rate-matched source.
7. **Throttled cap 256** (+ cap margin 1.25): a throttleable consumer whose producer is not at
   the pacer only needs short-timescale smoothing. mobilenet only: 22 edges held at 256
   (e.g. 7656 -> 256), total 64369 -> 9432. Not simulated (no mobilenet ZCU104 rtlsim). The
   margin changes one resnet50 edge (272 vs 256).
8. **Joins / reconvergent branches**: fill-frame peak, never relaxed (deadlock argument).
   resnet50 only: without it the total drops 362655 -> 54433 (shortcut 17405 -> 59).
9. **Idle-window floor** (branch edges only): tokens arriving while the consumer is outside its
   read window. resnet50 only: DWC -> FMPadding edges 1666-3332 vs 2 without; 362655 -> 330410.
10. **Absorbed-frame floor**: consumer TAV reads nothing (FINNLoop) -> one folded frame. Not
    exercised.
11. **Minimum depth 2, io_fifo_depth 5**: explains every case where the heuristic uses more
    storage than the reference on the MLPs (tfc 18.3 vs 9.1 kbit, cybersecurity 3.7 vs 1.9 =
    the 3136/600-bit input FIFO at 5 instead of 2).

## Regression set and results

Characterized fixtures (model + TAV sidecars + period, captured by intercepting
`DeriveFIFOSizes` inside the real `step_set_fifo_depths`): tfc-w1a1, tfc-w2a2, cnv-w1a1,
cnv-w2a2, cybersecurity-mlp, gtsrb, kws, vgg10-radioml, mobilenet_v1 (ZCU104) and resnet50
(V80 folding; the Vivado-version check bypassed in a scratch runner since nothing is
synthesized in quick mode). Baselines come from the unmodified 506ac5832 module. Fixtures
and JSONs: `$FINN_BUILD_DIR/fixtures/<model>/{characterized.onnx,tavs/,baseline.json,...}`.

* Refactor (3ab5b5101) and the three removed rules (8d0248e56): **identical depths on all
  10 fixtures** (edge by edge).
* Ablation sweep of every rule on all fixtures: `scratch/ablation_summary.txt` (first 9
  models) and `fixtures/resnet50/abl3_*.json`; summarized in the docs.
* rtlsim (ZCU104, stitched IP, 1 frame; reference = `largefifo_rtlsim`):

| model | reference fps / kbit | heuristic fps / kbit | ablations (fps / kbit) |
|---|---|---|---|
| tfc-w1a1 | 12.5 M / 9.1 | 12.5 M / 18.3 | (all depths minimal) |
| cnv-w1a1 | 12255 / 22.3 | 12255 / 11.9 | no burst floor 12077 / 9.7; no frame split 12255 / 18.4; no input pacing 12255 / 25.5 |
| kws | 25471 / 0.9 | 25471 / 4.3 | no burst floor 25471 / 0.9 |
| gtsrb | 5315 / 24.0 | 8134 / 9.9 | no slack 8134 / 10.8 |

On gtsrb the one-frame reference under-sizes two bursty consumers' FIFOs (DWC_13 -> MVAU_8: 2 vs 215, DWC_7 -> CIG_5: 2 vs 151) and is slower than the heuristic sizing.

The main sweep's rows (coordinator): identical throughput for all three strategies on
tfc/cnv/cybersecurity on ZCU104 and U55C; the storage differences are explained per FIFO in
the docs ("Comparison with the simulated reference"): io_fifo_depth on the MLPs, the paced
source on cnv (reference 617 vs 60 on the first FIFO).

## Harness changes (commit f7ca3e9a7)

`FEATURE_FLOW_MODELS` / `FEATURE_FLOW_BOARDS` env selection, `EXTRA_FEATURE_MODELS`
(tfc-w2a2, cnv-w2a2, gtsrb, kws, vgg10-radioml, mobilenet_v1, resnet50), custom-step
imports from the benchmark dir, golden pair from a custom streamline checkpoint.

## Open questions for the user

1. **Two passes vs one.** Only the causal pass is needed on every linear model (the non-causal
   pass never exceeds it there), but on resnet50 the non-causal pass is larger on 7 edges,
   including residual shortcuts (14106 vs 11870 tokens). If a resnet50 rtlsim with the
   causal-only depths holds throughput, the pass can be halved (and the code loses the
   `causal` parameter). Depth JSONs: `fixtures/resnet50/{baseline,only_A}.json`.
2. **Cap margin.** Changes exactly one depth in the whole set (resnet50 Pool_hls_1 ->
   MVAU_rtl_0, 272 vs 256). Worth keeping a constant for?
3. **Burst floor under a saturating source.** kws shows 3.4 kbit spent on a FIFO the
   reference does not need, because the sizer models a rate-matched source. Should the
   source model be a build option (saturating vs paced)? The paced assumption is also what
   keeps cnv-w1a1's first FIFO at 60 instead of 617.
4. **Idle-window floor only on branch edges.** It is the rule behind the deep
   DWC -> FMPadding FIFOs in resnet50 branches (1666-3332). It is not applied on chain
   edges by construction (inherited from the parent branch's `relaxed`, where the
   `burst_floor` argument was only read on the join path). Intended?
5. **Mobilenet and resnet50 throughput.** The cap, the pacer tolerance/reset and all join
   rules only have examples on mobilenet_v1 and resnet50, whose stitched-IP rtlsim is not
   available on this machine (ZCU104 URAM init issue; V80 needs Vivado 2024.2). Those
   examples are depth-only.

## Scratch tooling (not committed, in the session scratchpad `tools/`)

`characterize.py` (fixture capture), `size_fixture.py`, `compare.py`, `regress.sh`,
`make_ablations.py` + `ablate.sh` (one scratch module per ablated rule), `ablated_config.py` +
`run_variant.py` + `throughput.sh` (full-mode rtlsim of ablated depths through the harness
with `auto_fifo_depths=False`), `compare_sweep.py` (per-FIFO reference vs heuristic),
`quick_resnet50.py`.
