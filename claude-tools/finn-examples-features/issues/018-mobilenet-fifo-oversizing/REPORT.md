# Report 018: analytical FIFO sizing over-sizes MobileNetV1 — fixed for the committed folding

## Summary
* **Cause:** on the 15 largest edges the depth is the never-block peak occupancy. The slack relaxation gives that peak up (allowance 3079 to 24711 tokens against peaks of 77 to 259), but the **burst floor** puts it back. `min(peak, burst_above_rate)` equals the peak itself wherever the consumer reads fast relative to the frame rate. Every fast node in a chain runs a row ahead of the slow node below it, so the same backlog appears again at each such edge (12 of 15 edges). On the other 3 the **throttled cap** (256) sets the depth (peaks 259 to 425).
  * No other rule is involved: all four passes (causal × latency) give the same depth, except `FMPadding_0 -> SWG_1` (202 vs 193). The refill floor is 0, there are no joins/branches, IO FIFOs don't matter, and no DWC is sized on both sides (the narrow DWC outputs are all at 2).
* **Real need (full folding, stitched-IP rtlsim):** every FIFO into a throttleable consumer at 2 costs 0.8 %. Keeping only the three FIFOs right below the pacer (14 + 256 + 80 words) restores 788544 cycles exactly with **17.0 kB**, against 97.2 kB. That is the estimate (upper bound) of the optimum for our folding; LiveFIFO reaches ~22.1 kB on a 2x faster folding.
* **Fix** (`fix/018`, on `feature/dev/analytical-fifo-sizing`), rule "Decoupled chain": no burst floor or cap for a throttleable consumer when both of these hold:
  * the slowest node between the last pacer and the producer (`t_above`) takes at least 1 − 1/CAP_MARGIN (20 %) of the period;
  * `t_above` plus the slowest node at or below the consumer (`t_below`) take at most CAP_MARGIN (1.25) periods.
  * Result: full 97.2 → **17.0 kB** (identical to the optimum configuration above); small 64.9 → 64.9 kB (unchanged).

## 1. Which rule sets the depth
Full folding, offline (`scripts/explain.py`), period 788544.

| edge (width) | depth | rule | peak | slack allowance | burst |
|---|---|---|---|---|---|
| VVAU_2 -> DWC_10 (640) | 100 | burst floor = peak | 100 | 3079 | 150 |
| VVAU_4 -> DWC_18 (352) | 173 | burst floor = peak | 173 | 3079 | 458 |
| FMPadding_2 -> SWG_3 (256) | 206 | burst floor = peak | 206 | 3303 | 322 |
| MVAU_2 -> DWC_8 (208) | 226 | burst floor = peak | 226 | 18702 | 375 |
| VVAU_6..10 -> DWC (176), 5 edges | 223 | burst floor = peak | 223 | 3079 | 1168-2922 |
| MVAU_4 -> DWC_16 (208) | 176 | burst floor = peak | 176 | 6158 | 559 |
| FMPadding_4 -> SWG_5 (128) | 256 | throttled cap | 425 | 3535 | 738 |
| FMPadding_1 -> SWG_2 (128) | 256 | throttled cap | 259 | 24711 | 455 |
| VVAU_1 -> DWC_6 (352) | 80 | burst floor = peak | 80 | 5817 | 121 |
| VVAU_0 -> DWC_2 (320) | 82 | burst floor = peak (consumer drives the pacer) | 82 | 6160 | 204 |
| FMPadding_0 -> SWG_1 (128) | 202 | burst floor = peak (consumer drives the pacer) | 202 | 6384 | 249 |

* Every consumer can be throttled except the last two, which sit above the pacer (Thresholding_rtl_2 / DWC_4 / DWC_5).
* The sizer's peaks match an unbounded-FIFO behavioural rtlsim closely (VVAU_2 98, VVAU_6..10 209, MVAU_2 226). The chained schedule gets the never-block backlog right; the floor is what's wrong.
* Small folding: the big edges are `FMPadding_6..12 -> SWG` (6141 to 14565 words at 8 or 4 bit), set by the burst floor. The allowance is below the peak because the `MVAU_hls` nodes run at 0.90 of the period.

**Code-reading points from the write-up agent, checked on mobilenet:**
1. Joint two-frame evaluation counting peaks twice: not used. All 139 edges take the split path (`scripts/check_split.py`); joint evaluation would double them, e.g. 223 → 446.
2. Graph-input pace missing from the chain periods: only matters above the first pacer, where the burst floor sets the depths anyway. Mobilenet is unaffected.
3. Idle-window floor on join/branch edges: mobilenet has no joins.

## 2. Real need
Stitched-IP rtlsim with `scripts/perf.py`: real `fifo.sv`, 2 frames, interval in cycles.

**Full folding** (as sized: 788544 cycles, 97.2 kB; unbounded behavioural FIFOs also 788544):

| change (others as sized) | interval | kB |
|---|---|---|
| every FIFO into a throttleable consumer at 2 | 794807 (−0.8 %) | 9.5 |
| same set capped at 16 / at 48 | 793596 / 791004 | 17.5 / 33.6 |
| the 15 largest at 2 | 794519 | 24.4 |
| 3 FIFOs into pacer-driving consumers at 2 (FMPadding_0→SWG_1, VVAU_0→DWC_2, SWG_1→VVAU_0) | 791100 | 90.9 |
| FMPadding_0→SWG_1 alone at 2 / VVAU_0→DWC_2 alone at 2 | 788544 / 788544 | |
| the three right below the pacer at 2 (DWC_5→FMPadding_1 14, FMPadding_1→SWG_2 256, VVAU_1→DWC_6 80) | 791868 | 89.7 |
| FMPadding_1→SWG_2 alone at 2 / VVAU_1→DWC_6 alone at 2 | 789271 / 790430 | |
| VVAU_2..10 → DWC at 2 | 788544 | 56.8 |
| MVAU → DWC at 2 / DWC → MVAU at 2 | 788544 / 788544 | 86.9 / 92.6 |
| FMPadding_2..12 → SWG at 2 | 788544 | 72.5 |
| **all throttleable at 2 except the three right below the pacer** | **788544** | **17.0** |

**Small folding** (as sized: 897407 cycles, 64.9 kB; unbounded behavioural 888930 = estimate):

| change | interval |
|---|---|
| every FIFO capped at 2 / 16 / 128 | 1476659 / 1450635 / 1184639 |
| every FIFO into a throttleable consumer at 2 | 1476659 |
| DWC → MVAU_hls at 2 | 1265837 (−29 %) |
| FMPadding_6..12 → SWG at 2 | 913663 (−1.8 %, 17.7 kB) |
| FMPadding_1..5 → SWG at 2 | 898157 (−0.08 %) |
| DWC/Thresholding → FMPadding at 2 | 899842 (−0.27 %) |
| 3 pacer-driving FIFOs at 2 (DWC_0→MVAU_0 270, FMPadding_0→SWG_1 1624, DWC_2→FMPadding_0 137) | 897407 (unchanged, saves 4.2 kB) |

* With peak monitors on the FIFOs at their sized depths, `FMPadding_11 -> SWG_12` (a stride-2 SWG) never holds more than 7 of its 12742 words: 6.2 kB unused.
* Trimming every FIFO that never fills to its peak + 1 gives 57.6 kB.
* The small folding's optimum is not established: its MVAU_hls at 0.90 of the period leave little slack, and every cheap configuration above loses at least 0.08 %.

## 3. Fix
* **Code** (`derive_characteristic.py`): `downstream_periods()` returns the slowest period at or below each node. `edge_depth(..., t_above, t_below)` drops the floor for a throttleable consumer when `CAP_MARGIN * t_above >= (CAP_MARGIN - 1) * period` and `t_above + t_below <= CAP_MARGIN * period`. `t_above` is the slowest node between the pacer and the producer, taken from the producer's input edges' `t_prop`.
* **Docs:** new section "Decoupled chain" in `docs/finn/fifo_sizing_heuristics.rst`, plus a pointer from "Throttled cap".
* **Test:** `test_decoupled_chain_floor` (edge_depth cases and `downstream_periods` on the MLP chain).
* **Commits:**
  * `e3723ec90`: first version, with a consumer-side slack test.
  * `6f734bc6e`: the pair test above. The first version dropped `test_fifosizing_linear[cnv]` (target 1000 fps, MVAU_hls_6 and _7 both at 0.67 of the period) to 0.83 of the estimate by setting DWC_12→MVAU_7 from 63 to 20. Now that edge keeps 63 and the test passes.
* **Why the conditions:**
  * The three edges right below the pacer (only fast nodes in between) hold the pacer's continuous output and are needed.
  * Two slow stages on either side of an edge stall each other behind a shallow FIFO: small at 0.90 + 0.90, cnv-linear at 0.67 + 0.67.
* **Rejected variants:**
  * Floor minus the consumer's downstream slack: small at 0.64x in rtlsim.
  * Floor minus the cover held by the FIFOs above (`upstream_cover`): small fails, gtsrb −3.7 % in rtlsim.
  * Drop the floor whenever allowance ≥ CAP_MARGIN × peak: full −0.8 %.

**Offline re-sizing of all saved TAVs** (`scripts/compare_all.sh`, base = feature branch, total kbit):

| model | base | fix/018 |
|---|---|---|
| mobilenet_v1 (full) | 777.3 | 135.9 |
| vgg10-radioml | 15.6 | 15.3 |
| resnet18 | 2237.7 | 2225.2 |
| resnet50 | 2665.5 | 2664.6 |
| transformer_radioml | 387.0 | 386.2 |
| transformer_vision | 1291.1 | 1290.9 |
| espcn | 117806.4 | 117790.6 |
| bert | 7279.1 | 7277.6 |

Unchanged: bnn-pynq (9 models), cybersecurity-mlp, gtsrb, kws, cnv-w1a1, transformer_benchmark, transformer_language, yolov8 (2), mobilenet small.

**rtlsim spot checks** (same build, feature-branch depths vs fix/018 depths):

| model | base | fix/018 |
|---|---|---|
| vgg10-radioml | 1027 | 1027 |
| transformer_radioml | 2304 | 2304 |
| transformer_vision | 65535 | 65535 |

* espcn and resnet18 were still running at hand-back (`fx/fix018/res7`).
* Not simulated: resnet50 (V80, no rtlsim here) and bert (one edge 16 → 8 at 192 bit; hours of rtlsim).
* `test_fifosizing_linear`: tfc passed (first version), cnv passed (final version).

## Verification
Feature tests on `fix/018-int` (= test2/fifo + fix/018), `test_mobilenetv1_fifo_sizing[U55C-heuristic_analytical-*]`, logs in `fx/fix018-int/logs`:

| test | result | rtlsim / estimate | verified | FIFO storage |
|---|---|---|---|---|
| full | PASSED | 1.000 (788544) | yes | 136.1 kbit = 17.0 kB (was 97.2 kB) |
| small | PASSED | 0.9906 | yes | 64.9 kB |

The small test ran with the first commit (one 4-bit edge 17 → 2). The final rule gives exactly the original small depths, which also passed before at 0.9906.

## Open points
* The small folding's large `FMPadding -> SWG` FIFOs (47 kB in all) cost only 1.8 % at depth 2, and `FMPadding_11 -> SWG_12` (stride 2) is almost unused. An intermediate depth or a stride-2 SWG model could save much of the 64.9 kB; not done.
* The 20 % and 1.25 thresholds reuse CAP_MARGIN and are calibrated on mobilenet U55C plus the cnv linear test.

## Files
* Scripts in `scripts/`:
  * `gen_tavs.sh`, `tav_model.py`: TAV models.
  * `explain.py`, `periods.py`, `check_split.py`, `edge_trace.py`: which rule sets each edge's depth.
  * `eval_rule.py`, `variant_depths.py`: what-if rule variants.
  * `prep_copy.py`: copies a build's IP so diagnostic re-stitches never touch the original build dirs.
  * `perf.py`, `run_perf.sh`: re-simulation with chosen depths (`MON=1` bounded peak monitors, `BEHAV=1` unbounded FIFOs).
  * `peaks.py`, `sat.py`, `fifo_groups.py`, `cap_group.py`, `group_json.py`, `list_fifos.py`: depth sets and inspection.
  * `size_offline.py`, `compare_all.sh`: offline totals.
  * `run_tests.sh`: the feature tests.
* Data: `fx/fix018/{tav, res1..7, cmp*, depths}`; test builds in `fx/fix018-int`.
* Branches and worktrees (nothing pushed):
  * `fix/018`: `build/wt/fix018`
  * `fix/018-int`: `build/wt/fix018-int` (= test2/fifo + fix/018)
