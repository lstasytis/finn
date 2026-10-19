# Plan: make the chained-TAV FIFO sizer understandable

Worktree `/tmp/finn_dev_lstasytis/worktrees/fifo-clarity`, branch
`feature/dev/analytical-fifo-sizing-clarity`. Scratch tools in
`$SCRATCH/tools` (SCRATCH = the session scratchpad dir).

## Heuristics found in `derive_chained_tav_depths` / `DeriveFIFOSizes` (as of 55b0b08ce)

Core (not heuristics): per-node local schedule from the TAV; max-plus clock
`R(c) = max(R(c-1)+1, arrival of tokens read by c)`; depth = peak of
written - read on the resulting wall-clock schedules, periodic over the global period.

1. **Global period** = max(caller's estimate, slowest measured TAV period, 1).
2. **Graph-input pacing**: the graph input delivers one frame per global period,
   evenly spaced (otherwise the source is infinitely fast and every FIFO above the
   bottleneck grows without bound).
3. **Transparent nodes**: forks, joins, ElementwiseAdd* and nodes without TAV forward their
   input timeline with zero latency (join = max of inputs; fork = copy).
4. **Two-frame split of the peak** (`peak_occupancy`): fill frame vs steady frame
   evaluated separately; chain edges take the max; joins / nodes inside a reconvergent
   branch (`fill_only`) take the fill frame only.
5. **Causal writes** (`causal_writes`, second pass of `DeriveFIFOSizes`): on compacting
   nodes, output token j of the first frame is held until input ceil((j+1)*N_in/N_out)
   arrived; the pass is run with and without it and the per-edge max is kept.
6. **Pacer tolerance** (0.995): a node within 0.5 % of the slowest period counts as
   the pacer (the bottleneck is usually several nodes wide).
7. **drives_pacer**: a node at the pacer period or feeding one cannot be throttled.
8. **chain_period / t_up**: longest period in the chain feeding an edge since the last
   pacer; past a pacer "no constraint" (0) is propagated.
9. **Slack relaxation**: a producer chain finishing a frame in t_up < global period can be
   blocked for the difference, so `per_frame * (1 - t_up/global_period)` tokens are
   deducted from the peak.
10. **Burst floor**: depth >= min(peak, largest excursion of the consumer's demand above
    the edge's average rate line).
11. **Throttled cap** (256, SRL limit) with **cap margin** 1.25: for a throttleable
    consumer whose producer chain is not at the pacer and whose allowance >= 1.25 * peak,
    the burst floor is capped at 256.
12. **Join / branch edges are never relaxed** (deadlock, not slowdown); depth = max(peak,
    idle floor).
13. **Idle-window floor**: tokens arriving while the consumer is outside its read window
    (`per_frame * (1 - duty)`); capped at 256 for throttleable consumers.
14. **Absorbed-frame floor**: a consumer whose TAV reads nothing in the window (FINNLoop)
    gets one folded input frame.
15. **QUIRK input_rows_collapsed**: a node whose input[0] has a producer (through one DWC)
    only ever uses row 0 of its input TAV.
16. **QUIRK inherited_fork_join_tav_lengths**: forks and joins borrow a neighbour's TAV
    length as their own period (feeds chain_period / drives_pacer).
17. `DeriveFIFOSizes`: minimum depth 2, graph I/O >= io_fifo_depth (5), the max of the
    causal and non-causal pass.

## Regression set

Characterized fixtures (model with TAVs + the period handed to `DeriveFIFOSizes`),
captured by running the real `step_set_fifo_depths` with `DeriveFIFOSizes.apply`
intercepted (tool `characterize.py`); the sizer then runs on a fixture in well under a
second (tool `size_fixture.py` -> JSON `{producer -> consumer: depth}`).

Models (ZCU104, heuristic_analytical, quick mode of the feature-flow harness):
tfc-w1a1, cnv-w1a1, cybersecurity-mlp (cached already); add gtsrb, kws, vgg10-radioml,
mobilenet_v1, cnv-w2a2, tfc-w2a2 (harness change in its own commit; cache builds with
verification off). resnet50 (forks/joins) if its quick build works on ZCU104; otherwise the
fork/join rules are only covered by the unit fixtures.

Baseline depths are taken from the unmodified code (commit 506ac5832) before any edit.

## Refactoring steps

- [x] R0 baseline fixtures + depth JSONs: tfc-w1a1, tfc-w2a2, cnv-w1a1, cnv-w2a2, cybersecurity-mlp, gtsrb, kws, vgg10-radioml, mobilenet_v1 (ZCU104) and resnet50 (V80 config, Vivado check bypassed in a scratch runner; the only fork/join model).
- [x] R1 (commit 3ab5b5101) split `derive_chained_tav_depths` into named steps: read TAVs, node periods and
      pacer, branch membership, schedule propagation (one function per node), edge depth
      (peak -> depth), with one small dataclass/dict of graph-level facts. Identical depths.
- [x] R2 (scratch/ablation_summary.txt) ablation of every heuristic on the fixture set (depth diff only). Heuristics that
      change nothing on any model are candidates for removal; those that change depths get
      a throughput experiment.
- [x] R3 deliberate simplifications, all with identical depths on the regression set: single causal pass (13d45c007), cap margin dropped, input-row collapse quirk dropped (89ebc8a33). Tried and REVERTED because resnet50 depths change and resnet50 cannot be rtlsim'd here: transparent = no TAV (unary ElementwiseAdd with a TAV becomes characterized: 8 edges change, 362655 -> 355459) and dropping the idle-window floor (many edges, e.g. DWC_91 -> FMPadding_rtl_12 1666 -> 2, total -> 323208).
- [x] R4 tests/fpgadataflow/test_fifosizing_chained_tav.py (synthetic tree-model graphs + arithmetic units) over small committed fixtures (if a cheap fixture can be
      made; else the quick-mode harness is the regression).
- [ ] R5 docs: `docs/finn/fifo_sizing_heuristics.rst`, module docstrings, link from index.

## Experiments for the documentation examples

Per heuristic: ablate (scratch flag), size the fixture, list the edges that change, then
rtlsim both variants on the model where the change is largest. Throughput via the
feature-flow harness in full mode (`auto_fifo_depths=False` + a folding JSON carrying the
ablated depths, so the code under test is never modified for a measurement) against the
`largefifo_rtlsim` reference run. Models for full-mode runs: tfc-w1a1, cnv-w1a1 (IP cache
pre-seeded), plus gtsrb/kws if a heuristic only shows there.

## Validation

- Depth JSON identical on every fixture after each refactoring commit.
- Full mode (stitched IP + rtlsim) throughput >= 98 % of `largefifo_rtlsim` on tfc-w1a1 and
  cnv-w1a1 (and any model where depths changed deliberately), total FIFO kbit not worse.
- `tests/fpgadataflow/test_fifosizing.py` not run on this machine (AUP-ZU3 config check
  fails on Vivado 2023.1); unit tests marked `node_tree_modeling` untouched.
- Lint: isort / black -l 100 / ruff on every touched file.

## Log

- 2026-10-10 01:05 read the material, wrote this plan.

## Log (continued)

- 01:10 fixtures captured (characterize.py intercepts DeriveFIFOSizes inside the real step;
  TAV sidecars copied because cleanup_characterization unlinks them).
- 01:20 refactor identical on 8 fixtures; committed. mobilenet (141 edges) and resnet50
  (V80 config; needs the Vivado version check bypassed) added later.
- 01:30-07:00 (session interrupted) ablation sweep: no effect on any model: cap margin,
  fill_only, idle floor, join rule, measured period, both QUIRKs, transparent EWAdd,
  non-causal pass (only_causal identical; no_causal differs by 1 token on 3 edges).
  Effects: burst floor (biggest: cnv 271->176, kws 452->26, mobilenet 9432->3003),
  slack (cnv-w2a2 1405->1974, gtsrb 1298->1415), frame split (cnv 271->416), input
  pacing (cnv 271->849), cap / pacer tolerance / pacer reset (mobilenet 9432->~64k).
- rtlsim (ZCU104): cnv-w1a1 reference 12255 fps / 22.3 kbit; heuristic 12255 / 11.9;
  no_burst_floor 12077 (-1.45 %) / 9.7; no_frame_split 12255 / 18.4; no_input_pacing
  12255 / 25.5. kws: reference 25471 / 0.9 kbit; heuristic 25471 / 4.3; no_burst_floor
  25471 / 0.9 (the burst floor over-sizes Thresholding -> MVAU_0 to 428: the sizer assumes
  an evenly paced source, rtlsim's source saturates). tfc: 12.5 M fps both, 9.1 vs 18.3
  kbit = io_fifo_depth 5 vs 2 on the 3136-bit input only.
- Surprise: on cnv-w1a1 every nontrivial depth equals the peak occupancy; the slack
  relaxation never wins there (it does on cnv-w2a2 / gtsrb).
- 07:30 gtsrb full runs (reference, heuristic, no_slack) started; resnet50 fixture being
  baselined.
- 07:30 resnet50 is the only model where the join/branch rules, the fork/join period
  inheritance, the transparent-ElementwiseAdd rule and the idle-window floor can change a
  depth; ablations of all rules on resnet50 running (resnet_check.sh, 4 in parallel).
- 07:50 resnet50 (clean, two-pass module): no effect: transparent ElementwiseAdd rule,
  fork/join period inheritance -> both removed (identical on all 10). Effects, kept:
  idle-window floor (DWC -> FMPadding 1666-3332 vs 2; total 362655 -> 330410), join rule
  (-> 54433), fill_only (-> 378030), non-causal pass (7 edges, -> 355475), cap margin (one
  edge 272 vs 256). The single-pass and no-margin commits were therefore dropped from the
  history (git reset, nothing pushed); final commits: harness, refactor, three removed rules,
  unit test, docs.
- Final regression: identical depths on all 10 fixtures; unit test 3 passed.
- Open: gtsrb rtlsim (reference, heuristic 8134 fps / 9.9 kbit, no_slack variant) finishing.

## Task 2 (2026-10-12/13): impact ranking on Vivado 2024.2

- Depth ablations re-run with the final module on all 10 fixtures (abl4_*.json); cap-margin
  ablation corrected to a true removal of the guard (changes one resnet50 edge only).
- rtlsim 2024.2: reference + full + every depth-changing ablation on cnv-w1a1, cnv-w2a2, gtsrb,
  kws, vgg10-radioml. mobilenet_v1 fails in stitched-IP rtlsim (X output bits, as on 2023.1);
  resnet50 depth-only. Mistake on the way: the first cleanup pass deleted in-flight code-gen
  dirs of running cnv-w1a1/kws builds (re-run; cleanup now skips dirs younger than 3 h).
- Results in IMPACT.md and the docs section "Impact of each rule" (commit on the branch).
