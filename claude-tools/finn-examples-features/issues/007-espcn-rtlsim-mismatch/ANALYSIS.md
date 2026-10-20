# Issue 007 — espcn (KV260_SOM) stitched-IP rtlsim produces wrong outputs: analysis for a decision

Status 2026-10-20 ~13:00. Sources: `ISSUE.md` (task), `REPORT.md` (sessions 1-3, full evidence),
the tapped full-design simulations in the session scratchpad
(`/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/s/`).
**Root cause found (session 3):** an output-queue overflow in dev's MVU AXI wrapper
(`finn-rtllib/mvu/mvu_vvu_axi.sv`) when an `MVAU_rtl` runs at SF = 1 (SIMD = MW) under output
backpressure; in espcn that is MVAU_rtl_0. Sections 4-7 describe culprit, mechanism and options.

## 1. What fails

* Model: `espcn-bsd300` (super-resolution, upstream PR #1725, `tests/benchmark/espcn/`), board
  KV260_SOM. Custom steps lower the ConvTranspose to FMPadding_Pixel + Im2Col + MatMul.
* Tests: `test_espcn_fifo_sizing[KV260_SOM-heuristic_analytical]` (committed folding, analytic FIFOs)
  and `test_espcn_folding[KV260_SOM-target]` (optimizer folding at the committed throughput) fail
  stitched-IP rtlsim verification. `test_espcn_folding[KV260_SOM-maximize]` **passes**.
* Output (1, 3, 256, 256): ~10 % of the values differ from the golden output. Every wrong value is
  an exact 0 (except 10 values that are the known +-1/255 rounding spots), the pattern is the same in
  all three channels and data-independent: fixed column ranges in every output row, starting at a
  fixed row:

  | build | zero columns (of 256) | from output row |
  |---|---|---|
  | FIFO test (committed folding) | 188-201, 236-249; 0-9 | 76; 78 |
  | fold-target (optimizer folding, MVAU_3 SIMD 96) | 0-9; 104-123, 182-191 | 8; 16 |

## 2. What is correct (established)

* **The golden pair and FINN's Python flow are correct.** The streamlined, converted (HW layers,
  Python execution) and folded models reproduce the golden output within 1/255 (`scripts/pyexec.py`).
  Every non-zero value of the failing rtlsim output equals the Python HW-model output exactly. So the
  model, the #1725 custom steps and the golden I/O are fine; the bug is in the generated hardware's
  behaviour in rtlsim.
* **Every sub-graph and component is correct in isolation**, also under the real backpressure of the
  slow tail (OuterShuffle_hls, II=3): stitched rtlsim of the failing build's own IP from
  FMPadding_rtl_2 to the output with a free-running input (994k cycles), several smaller sub-graphs
  with paced inputs/outputs, and standalone random-handshake stress testbenches for `fifo.sv` (all
  depths of the design), `inner_shuffle`, and dev's DWC core `vpc.sv` (all 15 width pairs). Code
  review found no defect in the MVU wrapper/core, replay buffer, memstream, thresholding, parallel
  SWG, `input_gen.hpp`, FMPadding_Pixel or the XSI streamer (details: `REPORT.md` session 2).
  *Session 3 correction:* the MVU wrapper review missed the queue sizing (section 5), and no
  sub-graph containing MVAU_rtl_0 saw a burst-then-stall output pattern (the FIFO_0..FIFO_5 sub-sim
  used a regular 1-in-2 output throttle).
* **The FIFO depths should not matter for correctness.** FIFOs only change timing; a correct design
  produces the same values for any depth >= 2. The failure appears only in the full design, i.e. only
  with the timing that the full chain (head + tail) produces.

## 3. Throughput (separate, confirmed)

`OuterShuffle_hls_0` (from #1725's ShuffleDecomposition) reaches only **II = 3** in Vitis HLS
("Unable to enforce a carried dependence constraint" on `obuf_dat` in `input_gen`): 981,526 cycles
per frame vs its estimate 327,172, so the tail is the real bottleneck (~982k vs the 524k estimate)
and the whole pipeline runs backpressured from the tail. This is a real upstream (#1725/hlslib)
throughput bug independent of the wrong values, and it alone fails the tests' 0.98x throughput
check. It is also what puts the design into the backpressured regime in which the value bug shows.

## 4. Culprit: MVAU_rtl_0 (first conv layer), output queue overflow in dev's `mvu_vvu_axi.sv`

Tapped fold-target run with taps on every head FIFO (`H2_foldtarget_head_taps`, FIFO_0..FIFO_17,
95k cycles), compared word by word with the Python HW-model streams (`scripts/tapcheck.py`):

| FIFO (after node) | result |
|---|---|
| FIFO_1..FIFO_3 (FMPadding_0, SWG_0, DWC_0 = **MVAU_rtl_0's input**) | 0 mismatches, no X |
| **FIFO_4 (MVAU_rtl_0 output**, INT20 x PE 2) | **first X at word 28707, cycle 29791**; 80 X words of 81206, every other word exactly right |
| FIFO_5..FIFO_17 | X inherited (FIFO_5 34428, ..., FIFO_17 36281 = the X seen in section 4 of the previous version) |

* MVAU_rtl_0: MW 75, MH 64, **SIMD 75 (= MW, i.e. SF = 1: one complete dot product per cycle)**,
  PE 2, INT8 weights x UINT8, DSP48E2 core (`mvu.sv`). The committed folding and the fold-target
  folding both have SIMD 75; **fold-max has SIMD 25 (SF 3) and passes**.
* X words replace values one for one (no lost or extra words) and come in pairs of consecutive
  neuron folds (e.g. pixel 897 = row 7 col 1, folds 3+4); downstream nodes turn them into the
  "whole-pixel" X runs and finally the output zeros (resolved at the HLS ElementwiseMul).
* Same origin in the other builds: F2 (committed folding, FIFO test build) carries X from FIFO_20 at
  cycle 187672 (the tail's backpressure reaches MVAU_0 much later, hence errors from output row 76);
  G (behavioural MVAU_0/1 cores) also carries X -> not DSP-primitive related.
* Recorded backpressure just before the first X (Thr0 popping FIFO_4, depth 2): 1 word/cycle until
  29706, stall 11 cycles, 6 words, stall 61 cycles, 7 words (X at 29791), stall 4, X at 29796.
  MVAU_0's input is never the limit (it takes 1 input per 32 cycles, NF = 32).

## 5. Mechanism (confirmed with a standalone reproducer)

`finn-rtllib/mvu/mvu_vvu_axi.sv`, `blkOutput`: core outputs go into an SRL queue
`OBuf[0:MAX_IN_FLIGHT]` (pointer `OPtr`); on backpressure `OLock` stops feeding the core, and the
queue must absorb everything still in the core pipeline. `MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH =
3 + $clog2(SIMD+1) + (SIMD == 1)` (DSP48). Two errors:

1. **Stale latency formula.** Commit `e1561a40f` (2025-07-29, "Pipeline all levels with additive
   reductions") changed `mvu.sv` to `PIPELINE_DEPTH = 3 + $clog2(SIMD) + (SIMD == 1) + 1` but not
   the copy in `mvu_vvu_axi.sv`: for non-power-of-two SIMD the core is one cycle deeper than the
   wrapper assumes (SIMD 75: 11 vs 10). Still so on `upstream/dev`.
2. **Off by one in the queue size.** `OLock` is registered, so the item issued in the cycle in
   which the stall is detected also enters the pipeline: worst case L + 1 pushes without a pop
   (L = core latency), i.e. the queue needs L + 2 entries but has MAX_IN_FLIGHT + 1.

With SF = 1 a burst of full-rate outputs followed by a stall overflows the queue (SIMD 75 needs 13
entries, has 11; `OPtr` reaches 12): the oldest results are shifted out of `OBuf`, and the read
`OBuf[OPtr]` is out of range -> **X in simulation** (word count stays right, value lost). The
RTL's own assertion `Overflowing output queue` fires; FINN's XSI rtlsim does not surface `$error`,
so it went unnoticed. With SF >= 2 at most every second pipeline slot carries an output and the
queue never fills. On hardware the out-of-range read of the SRL is undefined (an SRL16/SRLC32 may
physically still hold the shifted-out value, so a board might be right by accident; not verified).
dev's own `mvu/tb/mvu_axi_tb.sv` has no SF = 1 configuration.

Reproducer `scripts/mvu_queue_tb.sv` (+ `mvu_queue_run.sh`): `mvu_vvu_axi` alone with MVAU_0's
parameters, inputs/weights always valid, every output checked against a dot-product reference.
50 k cycles each (full table in `REPORT.md` session 3):

| configuration | output tready | X words | OPtr max / queue entries |
|---|---|---|---|
| dev RTL, SIMD 75 (MVAU_0), behavioural / structural DSP | 50 % random | 8 / 7 | 12 / 11 |
| dev RTL, SIMD 75 | recorded fold-target pattern (replay) | 6 | 12 / 11 |
| dev RTL, SIMD 75 / 96 / 64 / 32 / 3 | bursty (stalls 1..12) | 508 / 479 / 207 / 244 / 1328 | over by 1-2 |
| dev RTL, SIMD 75, DSP48E1 / DSP58 | bursty | 455 / 472 | over |
| dev RTL, SF 2 or SF 5 | 50 % / bursty | 0 | 6 of 11, 1 of 8 |
| latency formula fixed only | random / bursty / replay | 1 / 165 / 2 | 12 / 12 |
| **latency formula fixed + `MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1`** | all of the above, SIMD 3/64/75/96, structural | **0** | L + 1 / L + 2 (tight) |

(The patched files are scratch copies; dev's RTL is unchanged.)

Not our branches: the FIFO, folding and DWC PRs only change timing and folding. The folding
optimizer can pick SF = 1 (it did for fold-target), as can hand-written foldings (the committed
espcn JSON). Any FIFO depth only moves the trigger point.

## 6. Status / next steps

* Localization done; all diagnostic simulations are finished or stopped (nothing running).
* If option A is chosen: apply the two-line fix on a dev-based branch, add an SF = 1 +
  backpressure case to `mvu_axi_tb.sv` (or ship `mvu_queue_tb.sv`), rerun the espcn fold-target and
  FIFO tests (full stitched rtlsim, 1-3 h each). Check the DSP58 depth formula the same way (the
  reproducer shows overflow there too; its value check for DSP58 SIMD 24 disagrees with my reference
  and is unresolved).
* Other campaign models with an `MVAU_rtl` at SF = 1 (MW = SIMD) under backpressure can show the
  same silent corruption; worth a grep of the final foldings when a model has unexplained rtlsim
  mismatches.

## 7. Options for you to decide

| option | what | effort / risk |
|---|---|---|
| A. Fix dev's RTL (recommended) | `mvu_vvu_axi.sv`: DSP48 `CORE_PIPELINE_DEPTH` = `mvu.sv`'s formula (or exported by the core), `MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1`; one SRL entry more; reproducer as test; separate PR to Xilinx/finn dev | needs your OK (dev RTL); two lines, verified standalone; full espcn rerun to close the issue |
| B. Report upstream only | issue on Xilinx/finn with the reproducer and this analysis; espcn FIFO/fold-target tests xfail referencing it | no RTL change by us; espcn stays red until upstream fixes it |
| C. Work around in the flow | forbid SF = 1 for `MVAU_rtl` in the folding optimizer, or change espcn's committed folding (MVAU_0 SIMD 25 as in fold-max) | hides a dev bug that hand foldings still hit; costs throughput/resources for layer 0; stopgap only |
| D. Fix the OuterShuffle II = 3 (independent) | hlslib `input_gen` carried dependence (or its estimate) | real throughput fix for #1725; would move, not remove, the value bug |

Recommendation: A (plus B as the upstream issue/PR description), D separately. Until A lands,
keep espcn's FIFO/fold-target tests out of the campaign and do not count them against the
FIFO/folding/DWC branches.

## 8. Evidence index

* Failing builds: `/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_espcn_KV260_SOM_aiguk3l6/`
  (committed folding), `fx/folding/fold-target_espcn_KV260_SOM_uvy3zp6x/` (optimizer folding),
  passing: `fx/folding/fold-max_espcn_KV260_SOM_nns10ty8/`.
* Tapped full-design simulations (scratchpad `s/`): `H2_foldtarget_head_taps/*.tap` (first X at
  FIFO_4), `H_foldtarget_full_taps/*.tap`, `F2_full_taps/`, `G_full_behavhead_taps/`; driver `scripts/subsim.py` (`SUBSIM_TAPS`,
  `SUBSIM_MODEL`, `SUBSIM_BEHAV_MVU`), checker `scripts/tapcheck.py`.
* The diagnostic FIFO tap is an uncommitted edit of `finn-rtllib/fifo/hdl/fifo_template.v` in
  worktree `build/wt/fix007` (to be reverted; never committed).
* Reproducer: `scripts/mvu_queue_tb.sv`, `scripts/mvu_queue_run.sh`, recorded sink pattern
  `scripts/fifo4_pops_foldtarget.txt.gz` (gunzip, pass as `+REPLAY=`).
* Stress testbenches: `scripts/fifo_stress_tb*.sv`, `inner_shuffle_stress_tb.sv`, `vpc_stress_tb.sv`.
