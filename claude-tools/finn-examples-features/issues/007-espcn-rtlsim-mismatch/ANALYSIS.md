# Issue 007 — espcn (KV260_SOM) stitched-IP rtlsim produces wrong outputs: analysis for a decision

Status 2026-10-20 ~12:30. Sources: `ISSUE.md` (task), `REPORT.md` (fix agent, sessions 1-2, full
evidence), the tapped full-design simulations in the session scratchpad
(`/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/s/`), and
the orchestrator's analysis of their tap logs (section 4, new). The root-cause node is not pinned
down yet; section 6 says what is running to pin it down and section 7 lists the options.

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

## 4. New today: the "zeros" are undefined (X) data from the head of the design

The fix agent's tapped full-design simulations log every word that passes selected FIFOs
(`<FIFO>.tap`: `cycle hexword`). In the fold-target run (`H_foldtarget_full_taps`) the taps contain
**X (undefined) values**; the agent's checker crashed on them (it parses hex), which is why session
2 concluded "the zeros are real RTL zeros" (true at the output, see below). Tap analysis (stream order
of the fold-target build):

| FIFO (after node) | words so far | X words | X pattern |
|---|---|---|---|
| FIFO_17 (MVAU_rtl_2 out; PE 4, SIMD 144, 32 ch) | 18354 | 1352 | first at word 5120 (cycle 36281); runs of exactly 32 words every 1024 words |
| FIFO_18..FIFO_24 (DWC, Thr2, DWC, FMPadding_Pixel, FMPadding_3, SWG_3) | | yes | propagated |
| FIFO_26 (MVAU_rtl_3 out), FIFO_27/28 (Thr3 in/out) | 23446 | 2178 | runs of 30 words every 768 words (from word 6144) |
| FIFO_29 (ElementwiseMul_hls_0 out) | 23442 | **0** | X resolved to concrete values |

Reading:
* MVAU_rtl_2's output stream is 8 words per pixel (32 channels / PE 4) and 1024 words per row of the
  128 x 128 feature map. **The X runs are the first 4 pixels (all channels) of every row from row 5**
  (later also other positions, e.g. a 64-word run at word 9640). Whole pixels across all output
  channels point at the *activations* entering MVAU_rtl_2, not at its weights (bad weights would hit
  specific channels of every pixel). Upstream of MVAU_rtl_2 are DWC_5 (64 -> 576),
  **ConvolutionInputGenerator_rtl_2** (SIMD 16, distributed-RAM line buffer, parallel_window 0),
  DWC_4, FMPadding_rtl_2, and FIFO_12 (depth 1255) after Thresholding_rtl_1.
* Row-start positions are where a sliding-window generator reads its line buffer across the row
  boundary. An X read from a distributed RAM means **an entry that was never written is read**: in
  simulation it is X; on the FPGA it would be the RAM's initial value (0) or stale data, i.e. wrong
  values in hardware too (not verified on a board).
* Downstream the X is only *resolved* at the HLS `ElementwiseMul_hls_0` (Thr3 passes X through as X;
  the HLS multiplier's RTL turns X operands into concrete values), which is why the output shows
  zeros at fixed columns and the 4-state readout never sees X.
* This explains why every sub-graph passed: the sub-sims start at FMPadding_rtl_2 or later with a
  free-running or regularly paced input, while in the full design the input of FMPadding_rtl_2 /
  SWG_2 comes from the head (MVAU_1, Thr1, DWC, a 1255-deep FIFO) with an irregular pattern, plus the
  tail's backpressure.
* Why only some foldings: the timing (FIFO depths, folding of SWG_2/MVAU_2) decides whether the
  generator's read overtakes its write. fold-max (more parallel, different buffers) does not trigger
  it; the committed and fold-target foldings do, at different rows/columns.

## 5. Where the fault most likely is (to be confirmed)

Ranked by the evidence:
1. **ConvolutionInputGenerator_rtl (dev's RTL SWG, default/non-parallel implementation,
   `finn-rtllib/swg/`)**: reads a line-buffer entry before it is written when its input stalls in a
   particular pattern while its output is backpressured. Row-boundary positions fit exactly. This would
   be a dev (upstream FINN) RTL bug. (Session 2 reviewed only the *parallel* SWG template, used by
   SWG_3; SWG_2 uses the default one.)
2. DWC_5 (`vpc.sv`, 64 -> 576) or DWC_4: stress-tested standalone without X; less likely.
3. FMPadding_rtl_2: emits the padding at row starts too; X there would also give row-start runs.
4. MVAU_rtl_2 itself (input buffering/replay): unlikely, whole-pixel pattern fits the input side.

Not our branches: none of these components is changed by the FIFO, folding or DWC PR branches. Our
branches only change the FIFO depths and foldings, i.e. which timing occurs. A correct design
tolerates any FIFO depth >= 2, so the analytic FIFO sizing is not at fault, but it (like any other
sizing) can expose the bug.

## 6. Running now / next steps

* The fix agent is rerunning the fold-target design with taps on FIFO_0..FIFO_17 (the head) to find
  the first FIFO that carries X and the cycle (~20-30 min of simulation: the first X appears at cycle
  ~36k). The node in front of that FIFO is the producer of the X.
* Then: a minimal reproducer (that node alone, driven with the recorded input/backpressure pattern
  around the first X), and the RTL mechanism (which memory or register is read before it is written,
  under which handshake). Result goes into `REPORT.md` "Session 3"; this file will be updated.
* The committed-folding run (F2) and the run with behavioural head MVAUs (G) are still running (taps
  FIFO_20+ / 23+ are clean so far; their errors start later, ~175k cycles).

## 7. Options for you to decide

| option | what | effort / risk |
|---|---|---|
| A. Fix the RTL bug at its source | after localization, a minimal fix in the faulty dev RTL module (e.g. SWG read/write pointer guard), with the reproducer as a test, as a separate dev PR | needs your OK (our rule: never modify dev's RTL without it); correct fix; upstream review needed |
| B. Report upstream only | open an issue on Xilinx/finn with the reproducer and the tap evidence; keep espcn FIFO/fold-target feature tests xfail (or skipped) referencing it | no RTL change by us; espcn stays red until upstream fixes it |
| C. Work around in the flow | e.g. avoid the triggering configuration for that op (another SWG implementation style / parallel_window, or FIFO depth around it) | hides a hardware bug that other designs can still hit; not recommended except as a stopgap |
| D. Fix the OuterShuffle II=3 (independent) | hlslib `input_gen` carried dependence (or its estimate in FINN) | removes the dominant backpressure; may make the value bug disappear in espcn without fixing it (another reason not to rely on it); a real throughput fix for #1725 |

Recommendation: finish localization (running), then A (or B if you prefer not to touch dev RTL),
plus D as a separate throughput fix. Until then, keep espcn's FIFO/fold-target tests out of the
campaign (done: they are excluded) and do not count them against the FIFO/folding/DWC branches.

## 8. Evidence index

* Failing builds: `/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_espcn_KV260_SOM_aiguk3l6/`
  (committed folding), `fx/folding/fold-target_espcn_KV260_SOM_uvy3zp6x/` (optimizer folding),
  passing: `fx/folding/fold-max_espcn_KV260_SOM_nns10ty8/`.
* Tapped full-design simulations (scratchpad `s/`): `H_foldtarget_full_taps/*.tap` (X evidence),
  `F2_full_taps/`, `G_full_behavhead_taps/`; driver `scripts/subsim.py` (`SUBSIM_TAPS`,
  `SUBSIM_MODEL`, `SUBSIM_BEHAV_MVU`), checker `scripts/tapcheck.py`.
* The diagnostic FIFO tap is an uncommitted edit of `finn-rtllib/fifo/hdl/fifo_template.v` in
  worktree `build/wt/fix007` (to be reverted; never committed).
* Stress testbenches: `scripts/fifo_stress_tb*.sv`, `inner_shuffle_stress_tb.sv`, `vpc_stress_tb.sv`.
