# Report 021: maximize-mode folding below the committed folding

Scripts in `scripts/` (synthesis, estimate dumps, fit, fast-loop probe, result collector),
data in `data/` (`measured_vs_estimated.md`, `mvau.tsv`, probe logs `old_*` / `v3_*` (0.8) /
`l75_*` (0.75) / `rom75_*` (final)). Work dirs: OOC synthesis `build/finn_build/fx/fix021/ooc/<tag>/`,
probes `build/finn_build/fx/fix021/probe/`. Worktrees: `build/wt/fix021-est`
(`feature/dev/resource-estimates`), `build/wt/fix021-fold` (`fix/021-fold`, fast-forwarded into
`feature/dev/set-folding-optimizer` in `build/wt/pr-folding`), `build/wt/fix021-old` (old
test2/folding tip `8a6e021e0`, "before" numbers), `build/wt/fix021-probe`, `build/wt/fix021`
(first experiments).

## 1. Summary

Not a search problem. On AUP-ZU3_8GB every model below 1.00x was bound by the **LUT budget
under FINN's LUT estimate**, which put the committed foldings at 1.5-12x the budget although
they are 7.5-53 k LUT (11-75 % of xczu3eg) in OOC synthesis. espcn (KV260) is different: its
committed folding **does not fit the board** (464 BRAM_18K for one OuterShuffle reorder buffer,
estimated 0 by FINN; the KV260 has 288).

Fix:
* new `feature/dev/resource-estimates` (off `upstream/dev` 11ebf1abb): MVAU_hls LUT and MVAU BRAM
  models refitted to OOC synthesis; MVAU_rtl LUTs (DSP48); RTL DWC LUTs; OuterShuffle BRAM/URAM.
  Per build the LUT estimate is now within -23..+13 % of synthesis (was -72 %..+7700 %).
* `feature/dev/set-folding-optimizer`: LUT budget 0.75 instead of 0.7 of the platform; resource
  view corrected by the weight values (+-1 weights stored as INT2 at one bit of memory; constant
  weights' LUTs scaled by their non-zero share); the verdict counts decomposed Shuffles.

Campaign (`v2__fold-max__*`, rtlsim/estimate >= 0.98 and stitched-IP verified in every passed job):

| model | board | before | after | fits (OOC synthesis / place & route) |
|---|---|---|---|---|
| tfc-w1a1 | ZU3 | 0.39x | **0.60x** (passed) | bound by LabelSelect PE=1 (`DO_NOT_FOLD`, fmax): 15 cycles vs 9 at the committed PE=5 |
| tfc-w1a2 | ZU3 | 0.43x | **1.00x** | (0.8 run, same 15 cycles: 52.6 k LUT synth, 48.0 k routed, 68 %) |
| tfc-w2a2 | ZU3 | 0.43x | **1.00x** | |
| cnv-w1a1 | ZU3 | 0.57x | 1.30x, **failed**: rtlsim 0.972 of its estimate (15482 fps = 1.26x the committed estimate) | 54.1 k LUT synth (77 %); the 0.8 run (1.73x) and the 0.75 run before the ROM-depth commit (1.54x) passed (1.000, 1.006): 54.6 k routed (77 %) |
| cybersecurity-mlp | ZU3 | 0.35x | **1.00x** (rtlsim 1.33x est) | = committed folding: 7.5 k LUT synth |
| gtsrb | ZU3 | 0.96x | **1.31x** (rtlsim 0.989) | (0.8 run, 1.71x: 53.2 k synth, 46.2 k routed, 65 %) |
| espcn | KV260 | 0.67x | 0.10x, verdict "does NOT fit (BRAM 485 > 230)" | job still in rtlsim after 5 h (5.2 M-cycle frames); no folding fits: the committed one is 468 BRAM_18K of 288 |
| cnv-w1a2 | ZU3 | 1.19x | 2.63x | 59.6 k LUT synth (84 %), 55.9 k routed (79 %) |
| cnv-w2a2 | ZU3 | 1.19x | 2.63x | 62.7 k LUT synth (89 %), 61.2 k routed (87 %), 324 BRAM_18K |
| cnv-w1a2 | KV260 | 1.79x | 4.77x (rtlsim 0.982) | 90.2 k LUT synth (77 %), 83.9 k routed (72 %) |
| lfc-w1a1 / lfc-w1a2 | ZU3 | 1.00x / 1.99x | 1.14x / 3.96x | |
| kws | ZU3 | 8.0x | 8.0x | |

## 2. Per-model diagnosis (old code; `data/old_*.log`)

Budget = 0.7 x 71000 = 49700 LUT, 0.8 x 412 = 329.6 BRAM_18K on ZU3 (platforms.py; the real
xczu3eg: 70560 LUT, 432 BRAM_18K); KV260: 81984 LUT, 230 BRAM_18K, 51 URAM, 998 DSP. "as built" =
the optimizer's verdict view (`_resources_as_built`, FINN estimates, without FIFOs).

| model | maximize: cycles, binding resource (share of budget) | committed: cycles, estimate as built vs budget | committed, OOC synthesis | problem |
|---|---|---|---|---|
| tfc-w1a1 | 23, LUT 31.6 k (64 %; the next faster step, 15 cycles, is 60 k) | 9, 230 k = 4.6x | 37.9 k (54 % of device) | estimate (and the LabelSelect rule caps at 15 cycles) |
| tfc-w1a2 | 35, LUT 39.3 k (79 %) | 15, 118 k = 2.4x | 52.0 k (74 %) | estimate, and budget (52 k > 49.7 k even if exact) |
| tfc-w2a2 | 35, LUT 39.2 k (79 %) | 15, 117 k = 2.4x | 49.8 k (71 %) | estimate, budget |
| cnv-w1a1 | 14407, LUT 49.1 k (99 %), BRAM 73 (22 %) | 8167, 162 k = 3.3x; BRAM 81 (real 302) | 43.7 k, 302 BRAM_18K | estimate (LUT 3.8x over, BRAM 3.7x under) |
| cybersecurity-mlp | 23, LUT 37.9 k (76 %) | 8, 585 k = 11.8x | 7.5 k (11 %) | estimate (91 % of the constant weights are zero) |
| gtsrb | 12807, LUT 48.5 k (98 %), BRAM 207 (63 %) | 12295, 76 k = 1.5x | 53.2 k (75 %), 180 BRAM_18K | estimate, budget |
| espcn | 786432, BRAM 217 (94 %: MVAU_rtl weights estimated in BRAM, LUTRAM in reality) | 524288, BRAM 1 ("fits": `distributed` weights estimated free, OuterShuffle not counted) | 91.7 k LUT, **468 BRAM_18K (device 288)** | reference infeasible; `distributed` excluded from the search as uncosted |

Wherever the committed folding fitted the old view, maximize already reached it (lfc, kws, cnv-w1a2,
fix 005), and here it could not reach the reference within the budget for any of the seven.

## 3. Measured vs estimated

No post-synthesis reports existed under `build/finn_build/fx/`. `scripts/ooc_synth.sh`: non-project
`synth_design -mode out_of_context` of a build's `stitched_ip/all_verilog_srcs.txt` (Vivado 2024.2,
2-4 threads, Xilinx float IP of espcn stubbed), `report_utilization -hierarchical`; `PNR=1` adds
`opt_design`/`place_design`/`route_design` at the build's 10 ns clock. 41 designs: committed
foldings (`fifo2/fifo-heuristic_analytical_*`, espcn `fifo/...`), old maximize and target
foldings (`folding2/fold-max_*`, `fold-target_*`) and the new maximize foldings, on xczu3eg and
xck26. Estimates (`scripts/node_est.py`, `node_est_opt.py`) of each build's
`step_create_stitched_ip.onnx` (with FIFOs and DWCs). Full tables: `data/measured_vs_estimated.md`.

### 3.1 Per build, LUT (estimate / measured)

| build | measured | old | new (optimizer view) |
|---|---|---|---|
| committed tfc-w1a1 / w1a2 / w2a2 | 37.9 k / 52.0 k / 49.8 k | 6.19 / 2.33 / 2.40 | 1.13 / 1.02 / 1.05 |
| committed cnv-w1a1 | 43.7 k | 3.78 | 1.08 |
| committed cybersecurity-mlp | 7.5 k | 77.9 | 2.47 (dense custom-op estimate 24.1) |
| committed gtsrb / kws | 53.2 k / 24.2 k | 1.50 / 0.62 | 1.04 / 0.91 |
| committed lfc-w1a1 / w1a2 | 18.6 k / 22.3 k | 2.25 / 1.29 | 0.96 / 0.90 |
| committed espcn (KV260) | 91.7 k | 0.28 | 0.90 |
| old maximize, 13 builds | 8.5-80.4 k | 0.28-4.56 | 0.82-1.10 (cybersecurity 2.46) |
| old target, 7 builds | 7.5-52.4 k | 0.27-77.9 | 0.77-1.03 (cybersecurity 2.47) |
| new maximize, 9 builds | 7.5-100.0 k | 1.6-24 | 0.86-1.10 (cybersecurity 2.47) |

BRAM_18K: committed cnv-w1a1 302 measured / 81 old / 301 new; gtsrb 180 / 315 / 179; espcn 468 /
10 / 466; lfc-w1a2 (+-1 weights) 176 / 316 / 240.

### 3.2 Per node type (36 builds, per-node attribution)

| op type | nodes | measured LUT | old est | new est | measured BRAM_18K | old | new |
|---|---|---|---|---|---|---|---|
| MVAU_hls | 197 | 1024462 | 3409176 | 1339796 (incl. 3 x 175 k dense cybersecurity, scaled down by the optimizer) | 3165 | 3499 | 4820 |
| MVAU_rtl (espcn) | 12 | 170359 | 0 | 162427 | 0 | 506 | 0 |
| StreamingFIFO_rtl | 746 | 121602 | 108611 | unchanged | 165 | 182 | 182 |
| ConvolutionInputGenerator_rtl | 116 | 51296 | 70502 | unchanged (1.37x) | 0 | 0 | 0 |
| Thresholding_rtl | 42 | 11740 | 17032 | unchanged (1.45x) | 19 | 0 | 0 |
| StreamingDataWidthConverter_rtl | 263 | 6306 | 50532 | 15224 | 0 | 0 | 0 |
| OuterShuffle_hls (espcn) | 3 | 3249 | 0 | 0 | 1392 | 0 | 1368 |
| LabelSelect, Pool, FMPadding, InnerShuffle | 74 | 6016 | 0 | 0 | 12 | 0 | 0 |

(Per node, an embedded ROM's BRAM can land outside its node's hierarchy, and Vivado builds some
ROMs of +-1 weights from LUTs, hence MVAU_hls BRAM over; per build totals in 3.1.)

### 3.3 What was wrong

MVAU_hls (fit: `scripts/fit_mvau.py`, 157 then 197 nodes, weighted least squares):
* compute LUTs: FINN-R's `1.1 ((2 ceil((W+A)/6) - 1)(W+A) + (W+A)(2-1/Q))` per lane gives 6.6 per
  XNOR lane (measured 0.9-1.5), 13.2 at 2x2 bits (measured 5-6), 55 at 2x8 (15). Now
  `500 + lanes (0.9 XNOR | 0.85 (W+A) + 0.36 W A) + PE 0.8 acc (1 + n_thr) + weight memory`;
* threshold comparators were only counted with `distributed` threshold memory (kws, 7 comparators
  of 18 bits per PE: 2.8x under);
* weight memory LUTs: embedded ROM of <= 16 words as logic (1.25 LUT/bit x depth/16), BRAM ROM read
  path 1.25 LUT/bit, decoupled stream 2.1 LUT/bit plus one SRL per bit plus LUTRAM when shallow;
* BRAM: embedded weights counted 0 up to 128 words; Vivado maps a ROM of > 16 words to BRAM, Vitis
  HLS makes one ROM per PE of power-of-two depth (cnv-w1a1's 32-word layers: 64 and 128
  BRAM_18K). Vitis HLS ignores `ram_style` for embedded weights (lfc-w1a1's "distributed"
  MVAU_hls_3 got 10 BRAM_18K). Decoupled `auto` of <= 48 words is LUTRAM (64: BRAM).

Weight values (custom ops cannot see them; corrected in SetFolding `node_resources`):
* +-1 weights become INT2 in `minimize_weight_bit_width` (issue 005); one bit of every word is
  constant and trimmed: lfc-w1a2 176 BRAM_18K measured, 426 estimated at 2 bits, 176 at 1;
* constant weights (fully unrolled, one word per PE): HLS drops zero-weight multiply-adds;
  cybersecurity-mlp has 91 % zeros (7.5 k measured, 181 k dense, 18.5 k scaled).

Others: MVAU_rtl LUTs were 0 (measured 18-40 per lane on DSP48E2; now 2.3 (W+A) per lane plus
LUTRAM weights; DSP58 left at 0, not measured); RTL DWC counted the wider stream's bits (7.9x
over; now the narrower + 30); OuterShuffle's reorder buffer (2^18 x 32 bit in espcn) was not
estimated (now by capacity, URAM above 262144 words as in its II model).

Residual error, where it matters: Vivado builds some weight ROMs from LUTs instead of BRAM
(+-1 weights, cnv-w1a2/lfc-w1a2: LUT 0.82-0.88x, BRAM over) and cnv-w2a2's wide 12-word ROM
logic (0.86x). Not recalibrated: SWG and Thresholding (over-estimated), LabelSelect/Pool/FMPadding
(0, ~100 LUT each), VVAU (no data in these models).

## 4. Changes

| branch | commit | file | +/- | what |
|---|---|---|---|---|
| `feature/dev/resource-estimates` (new, off `upstream/dev` 11ebf1abb) | a1e5906ac, 55c9bb8a2 | `custom_op/fpgadataflow/matrixvectoractivation.py` | +79 / -34 | `bram18_count`, `rom_in_bram`, `weights_in_luts`, `threshold_bram_estimation`; per-PE ROM BRAM; ROM > 16 words in BRAM |
| | | `hls/matrixvectoractivation_hls.py` | +32 / -47 | fitted `lut_estimation` |
| | | `rtl/matrixvectoractivation_rtl.py` | +14 / -1 | `lut_estimation` (DSP48) |
| | | `rtl/streamingdatawidthconverter_rtl.py` | +6 / -0 | `lut_estimation` |
| | | `outer_shuffle.py` | +36 / -0 | `reorder_buffer`, `bram_estimation`, `uram_estimation` |
| `feature/dev/set-folding-optimizer` (on the other agent's 69d005339) | 00951c1e7, f3e0dfade | `transformation/fpgadataflow/set_folding.py` | +92 / -9 | `RESOURCE_LIMITS` (LUT 0.75), `weight_info` / `node_resources`, Shuffle decomposition in `_resources_as_built` |
| `claude-tools` | | `tools/rebuild_branches.sh` | +5 / -4 | `EST` branch in every rebuild |
| | | `tools/status_notes.json` | +5 / -2 | #021 notes |

Landed: `feature/dev/set-folding-optimizer` fast-forwarded (worktree `pr-folding`) and merged
into `feature/dev/generalized-datawidthconverter` (850a69531); both feature branches merged into
`dev-finnexamples-features` (5d1afd299; OuterShuffle conflict with the FIFO branch's refactor:
both sides kept, rerere recorded); all four commits cherry-picked onto `test2/folding` (tip
1f64732c5) and `test2/dwc` (ad2a5a903). `tools/rebuild_branches.sh` merges the new branch into
every rebuild. Not pushed.

Tests: `tests/fpgadataflow/test_set_folding.py` 50 passed / 5 skipped / 1 xfailed on the folding
branch, 50 / 2 / 1 with the estimate changes; `test_fpgadataflow_res_estimate.py` and
`tests/util/test_estimate_only_no_vivado.py` (46) pass; black/isort/flake8 clean. Target mode
shares the estimates and the budget: target foldings get cheaper in LUT, fits verdicts change
where the old estimate was over budget.

## 5. Budget limit

LUT 0.75 x the platform's count (53250 on ZU3, 87840 on KV260); BRAM/URAM/DSP stay at FINN's 0.8.
* The committed foldings, built for these boards, use up to 75 % of xczu3eg's LUTs (gtsrb 53.2 k,
  tfc-w1a2 52.0 k): 0.7 rejects them even with exact estimates (tfc-w1a2: 49.8 k as built > 49.7 k).
* 0.8 was tried first: tfc-w1a2 74.5 %, gtsrb 75.4 %, cnv-w1a1 84 % post-synthesis, but cnv-w2a2
  (estimate 0.86x of synthesis) 98.5 % (95 % routed, 99.98 % of CLBs): too full for the board with
  its shell. At 0.75 that cnv-w2a2 folding is 88.9 % (86.8 % routed) and every affected model still
  reaches its reference (fast loop and campaign).
* Post-route LUTs are 3-16 % below post-synthesis (opt_design): committed gtsrb 53.2 k -> 44.8 k.
* The budget compares the as-built estimate without FIFOs; the FIFOs add 1-4 k LUT on these models.

## 6. Fit validation (OOC, 10 ns, xczu3eg-sfvc784-2-e / xck26-sfvc784-2LV-c)

| design | limit | LUT synth | LUT routed | BRAM tiles | CLBs | routed | WNS |
|---|---|---|---|---|---|---|---|
| committed gtsrb | - | 53181 (75 %) | 44816 (64 %) | 90 | | yes | +3.09 |
| maximize tfc-w1a2 (1.00x) | 0.8 | 52583 (75 %) | 47984 (68 %) | 0 | 93 % | yes | +0.30 |
| maximize gtsrb (1.71x) | 0.8 | 53171 (75 %) | 46185 (65 %) | 88 | 96 % | yes | +3.14 |
| maximize cnv-w1a1 (1.73x) | 0.8 | 59345 (84 %) | 56975 (81 %) | 42 | 99 % | yes | +1.24 |
| maximize cnv-w2a2 (3.06x) | 0.8 | 69509 (98.5 %) | 67091 (95 %) | 161.5 | 100 % | yes | +2.35 |
| maximize cnv-w2a2 (2.63x) | 0.75 | 62741 (89 %) | 61220 (87 %) | 162 | 99.9 % | yes | +2.37 |
| maximize cnv-w1a1 (1.54x, intermediate) | 0.75 | 56717 (80 %) | 54592 (77 %) | 103 | 99 % | yes | +2.29 |
| KV260 maximize cnv-w1a2 (5.15x) | 0.8 | 100019 (85 %) | 93503 (80 %) | 34.5 | 99 % | yes | +0.62 |
| KV260 maximize cnv-w1a2 (4.58x) | 0.75 | 95787 (82 %) | 89037 (76 %) | 38 | 97 % | yes | +0.91 |
| final cnv-w1a2 ZU3 (2.63x) | 0.75 | 59586 (84 %) | 55890 (79 %) | 45.5 | 99.9 % | yes | +2.35 |
| final cnv-w1a1 ZU3 (1.30x, rtlsim 0.972) | 0.75 | 54102 (77 %) | - | 38 | | | |
| final KV260 cnv-w1a2 (4.77x) | 0.75 | 90151 (77 %) | 83948 (72 %) | 32.5 | 97 % | yes | +1.02 |

All routed without errors and met 10 ns OOC (opt_design/place_design/route_design, no shell). On
ZU3 the final foldings leave >= 8 k LUT (routed) for the shell (DMA, interconnect); CLB occupancy is
high (97-100 %) because the placer spreads LUTs over all slices.

## 7. Open points

* cnv-w1a1's final maximize folding (1.30x) fails the 0.98 bar at 0.972: a very balanced folding
  (MVAU_hls_1 6279 cycles, the 8->16 bit DWC behind it 6272, several SWG/DWC/MVAUs at 5.1-6.0 k),
  rtlsim interval 6459. Not a resource problem (the issue-006 class: nodes paced at exactly the
  period lose a few %; the optimizer may need a cycle margin). The folding moved from 1.54x
  (passed) when the ROM-depth commit made a 128-PE shallow-ROM layer cost its real 128 BRAM_18K.

* tfc-w1a1 stays at 0.60x while LabelSelect is kept at PE=1 (`DO_NOT_FOLD`; 610 MHz at PE=1, 230 at
  PE=5): at a realistic clock the maximized folding may still be faster. User decision (HANDOFF).
* espcn on KV260: every folding needs the OuterShuffle's 8 Mbit reorder buffer (464 BRAM_18K of
  288); maximize now says so and returns the least-resource folding. As URAM (29 of 64) it would fit,
  but the HLS infers URAM only above 262144 words (then II=3). Upstream (Shuffle decomposition).
* Weight values are corrected only in SetFolding, not in FINN's estimate reports (+-1 weights as
  INT2, sparse constant weights). The BIPOLAR -> INT2 widening (issue 005) remains upstream.
* Other boards: MVAU_rtl LUTs (DSP48), OuterShuffle buffers and the 0.75 LUT limit also apply to
  U55C/ZCU104/V80/VCK190 models (transformers, resnet, yolov8, bert, mobilenet), whose fold-max
  jobs were not rerun here (fast loop: vgg10 1.00x and cnv-w2a2 U55C 20.97x unchanged).
* The DWC small jobs that failed at 01:22 (cnv-w1a2/w2a2 rtlsim 0.937 of estimate) started at 00:38
  with the old code; not caused by this change.
