# Report 007: espcn (KV260_SOM) stitched-IP rtlsim output differs from the golden — root cause (session 3): MVU output queue overflow in dev RTL
(Written by the orchestrator from the fix agent's hand-back, state 18:40; scripts in `scripts/`.)

## Q1: golden pair vs FINN Python execution: consistent
Reference build `est_espcn_KV260_SOM_uu6ftob8` on the golden input: `step_espcn_attach_preproc`,
`step_espcn_lower_deconv`, `step_convert_to_hw` (HW layers, Python) all within 1/255 (4e-5..6e-5 of
values, rounding; well inside atol). The #1725 flow and the golden are correct. Every non-zero value
of the failing stitched-IP output equals the Python HW-model output exactly.

## Q2: first diverging node: not found
* Every wrong value is an exact **0** (20460 values, 6820 per channel; the golden has no zeros);
  same pattern in all channels, data-independent: columns 188-201 and 236-249 zero in every row from
  output row 76, columns 0-9 zero from row 78.
* Exact equivalent (`zerotest.py`): zeroing, in the Thresholding_rtl_2 output (128x128, the
  FMPadding_Pixel input), pixels 93-101 and 117-125 from row 37 and pixels 0-5 from row 38 (all 32
  channels), then running the rest in Python reproduces the zero map exactly (6820/6820). So whole
  pixels are replaced by zeros from ~29 % into the frame. Thr2 maps accumulator 0 to non-zero values,
  so the zeros come out of Thr2 itself or appear later before SWG_3. Errors start mid-frame and repeat
  in the same columns -> a node misbehaving once backpressure reaches it.
* **Real throughput bug (upstream/#1725):** `OuterShuffle_hls_0` reaches only II=3
  (`vitis_hls.log`: "Unable to enforce a carried dependence constraint" on `obuf_dat` in
  `input_gen`, Final II = 3): 981,526 cycles/frame alone vs estimate 327,172 -> the tail is the real
  bottleneck (~982k vs the 524k estimate), the whole pipeline is backpressured from the tail, which
  the analytical sizing didn't expect. This alone would also fail the 0.98x throughput check.
* Sub-graph stitched-IP rtlsim with the failing build's IP (input/reference from the Python HW model)
  — **all correct (0 mismatches)**: DWC_14..FIFO_35 free (982k cycles); FMPadding_Pixel..FIFO_35 free
  + tail backpressure (989k); FMPadding_Pixel..FIFO_29 free / input 1-in-4 (530k/657k); DWC_9..FIFO_29
  Thr2-paced (531k); **DWC_9..FIFO_35 Thr2-paced + full tail backpressure (990k)**; FIFO_0..FIFO_5
  output 1-in-2 (1146k); FMPadding_rtl_2..FIFO_20 output 1-in-8 (1053k); DWC_2..FIFO_8 (URAM FIFO_7
  depth 2094) 4 throttle combinations (540k-1082k); Thr3, ElementwiseMul, InnerShuffle, OuterShuffle
  alone. Standalone `fifo.sv` stress tests at all espcn depths (1060, 1040, 510, 2094, 2912, 2190,
  5587, 513, 272, 256, 230) with random/phased valid/ready pass.
* Unfinished when stopped (18:32): head FIFO_0..FIFO_23 free and throttled, layer 1 with
  backpressure, full model with behavioural MVU cores.
* Other builds: `fold-target_espcn_KV260_SOM_uvy3zp6x` fails the same way (31920 zeros, fixed column
  stripes from row 8); `fold-max_espcn_KV260_SOM_nns10ty8` passes -> depends on folding/FIFO timing.

## Q3: where the fix belongs: open, evidence points away from our branches
Golden and Python flow correct, FIFOs correct in isolation, FIFO depths shouldn't change values.
Suspects (dev nodes misbehaving only under the full chain's timing): Thresholding_rtl_2, DWC_10,
FIFO_23, FMPadding_Pixel_hls (HLS `style=flp` pipeline with a conditional blocking read),
FMPadding_rtl_3, DWC_12. OuterShuffle II=3 belongs to dev/#1725 (HLS or its estimate).

## Evidence
Failing build `/home/lstasytis/finn/build/finn_build/fx/fifo/fifo-heuristic_analytical_espcn_KV260_SOM_aiguk3l6/`;
stitched projects `/home/lstasytis/finn/build/finn_build/fx/fix007/`. Worktree `build/wt/fix007`
(`fix/007` = `test/fifo` @ 8dcdd9a91, no changes).

## How to resume
1. Find the first diverging node inside the full design: full stitched rtlsim with `rtlsim_trace` or
   stream taps on FIFO_21..FIFO_27; compare Thr2's output stream from word index 154,528
   ((37*128+93)*32) with the Python reference; stop after ~600k cycles (~55 cycles/s at first: 1.5 h+).
2. Cheaper trigger test: full model with OuterShuffle replaced by a fast sink: if the zeros vanish,
   tail backpressure is the trigger.
3. Diagnostic only: deepen FIFO_23/FIFO_25 on the full model (`subsim.py ... FIFO=depth`): if values
   change, a node misbehaves under backpressure.
4. Fix OuterShuffle II=3 or its estimate upstream (may hide rather than fix the bug).
Scripts: `pyexec.py`, `subsim.py` (env `SUBSIM_THROTTLE`, `SUBSIM_OTHROTTLE`, `SUBSIM_BEHAV_MVU`),
`zerotest.py`, `fifo_stress_tb.sv`. `subsim.py`/`zerotest.py` need `ctx_hw.npz` next to them:
`python pyexec.py <est build>/intermediate_models/step_convert_to_hw.onnx ctx_hw.npz`.

## Session 2 (2026-10-20, 10:50-12:00, stopped by the harness while the decisive sims were running)

### Corrections to session 1
* **The wrong values are NOT whole Thr2 pixels.** Of the 20470 wrong output values only 10 are
  non-zero (the known +-1/255 rounding spots of Q1); every other wrong value is an exact 0, and
  output pixels whose 3x3 window only *partially* overlaps a "zeroed" region are exactly correct.
  Zeros before SWG_3 would smear through the 6x6 window, so the zeros are injected **after SWG_3's
  windowing**: whole MVAU_3 output words (PE=3 -> all 3 channels of a pixel) or later. Session 1's
  `zerotest.py` only matched the zero *map*, not the values. Thr3 has a single broadcast threshold
  set with T0 = 31, so any MVAU_3 accumulator < 31 (e.g. 0) gives output 0 -> Mul -> 0.0.
* Pattern restated at the output level: fifo build: columns 188-201 and 236-249 of every row from
  row 76, plus 0-9 from row 78 (runs of 14/14/10 pixels = 42/42/30 words at Thr3, at fixed
  positions inside each 256-pixel output row). fold-target build (`fold-target_espcn_KV260_SOM_uvy3zp6x`,
  MVAU_3 SIMD=96): columns 0-9 from row 8, plus 104-123 and 182-191 from row 16. fold-max passes.
* The rtlsim readout is 4-state (`as_hexstr`); X would crash `int(x,16)`, so the zeros are real
  RTL zeros. `verify_rtlsim_behavioral` defaults to False and the python rtlsim path derives `behav`
  from the `rtlsim_behavioral` metadata prop (unset in both the verification and the sub-sims), so
  the failing run and all sub-sims use the same DUT (structural DSP48E2 MVUs, real `fifo.sv`).
* Thr2's thresholds: 15 of 32 channels have T0 = 2048 (always 0), channel 1 has T0 = -13, so an
  all-zero Thr2 pixel needs accumulators < -139 in every channel; acc 0 does not give it.

### Ruled out this session (all 0 mismatches)
* Sub-graph stitched rtlsim, real IP of the failing build, real tail backpressure:
  **FMPadding_rtl_2 .. FIFO_35 free input (994090 cycles, 2901 s)** — i.e. SWG_2, MVAU_2, DWC_9,
  Thr2, DWC_10, FIFO_23, FMPadding_Pixel, FMPadding_rtl_3, SWG_3, MVAU_3, Thr3, Mul, InnerShuffle,
  OuterShuffle together are correct when the input to FMPadding_rtl_2 is free-running;
  FIFO_18 .. FIFO_23 (MVAU_2+Thr2+DWC_10) 524351 cycles; DWC_14 .. FIFO_35 with paced input
  `SUBSIM_THROTTLE` 1,7 / 1,15 / 3,40 (983k / 1377k / 1268k cycles).
* Standalone random-handshake stress testbenches (xsim, `scripts/*_stress_tb*.sv`): `inner_shuffle`
  (BITS 32, I 256, J 3, SIMD 1; 40 pages), `fifo.sv` at depths 2, 3, 5, 6, 35, 39, 230, 256, 272,
  513 (session 1 covered the deep ones), `vpc` (dev's DWC core) at all 15 width pairs of the design
  incl. 128->576 (generic path), 48->12, 4->16, 4608->576, 57->19.
* Code review, no defect found: `mvu_vvu_axi.sv` (output queue/OLock), `mvu.sv` structural vs
  behavioural paths (same L pipeline), `replay_buffer.sv`, `memstream.sv`, `thresholding(_axi).sv`,
  `vpc.sv` ser/des/generic paths, `fifo.sv` all three backings, `inner_shuffle.sv`/`elasticmem.sv`/
  `skid.sv`, `swg_template_parallel.sv` (pure lock-step), hlslib `input_gen.hpp` (OuterShuffle:
  circular buffer with 4-iteration write-pointer delay, `read_nb`/`write_nb`), `FMPadding_Pixel`,
  the XSI `InputStreamer` (holds data while valid && !ready).

### Where it stands
Every component and every sub-graph from SWG_2 downwards is correct in isolation and under the
real tail backpressure, but the full design fails -> the trigger needs the **head** (FIFO_0..FIFO_17:
SWG_0/1, MVAU_0 (8-bit, 75 DSPs), MVAU_1 (PE 8, two DSP pipes), Thr0/1, the 4.5:1 DWCs). A
fill-rate estimate (head 8 vs tail 9 cycles per output pixel; sum of all FIFO/line-buffer capacities)
puts the arrival of the backpressure wave at the global input at ~output row 70, close to the first
error (row 76); in fold-target the buffers differ and the error starts at row 8.

### Running when stopped (started 11:45-11:49, `setsid nohup`, PID files next to the logs)
All in `/tmp/claude-1205/-home-lstasytis-finn/3a5e3dcc-3aca-42a9-8340-515e8131d721/scratchpad/s/`
(scratchpad; copy `s/` somewhere persistent first if the container restarts):
* `F2_full_taps` (PID 1656574): **full fifo-build model, structural, with FIFO taps** on FIFO
  20,21,22,23,25,27,30,31,32,33,34,35 (~80 cycles/s, 1M cycles -> done ~15:30; first error expected
  at ~175k cycles, ~12:30).
* `H_foldtarget_full_taps` (PID 1769207): full **fold-target** model (fails from row 8 -> first
  error within ~20k cycles), taps FIFO 17,18,19,20,21,23,24,26,27,28,29,30,31 (its FIFO numbering:
  F23 = SWG_3 in, F24 = SWG_3 out, F26 = MVAU_3 out, F27 = Thr3 in, F28 = Thr3 out, F29 = Mul out).
* `G_full_behavhead_taps` (PID 1769285): full fifo-build model with behavioural MVAU_0/1 cores
  (cycle-identical AXI timing, fewer DSP primitives), taps FIFO 23,27,28,30,31,32,33,34,35.
Taps are `<run>/<FIFO>.tap`, one `cycle hexword` line per output transaction, flushed live.
`python tapcheck.py <run>/sub.onnx <run> [FIFO...]` compares each tap with the Python reference
packed to the stream width and prints the first mismatching word and its cycle; `tapwatch.sh <run>
<interval> <FIFO...>` loops it. Kill with `kill <PID>` (then `pgrep -P <PID>`), never `pkill -f`.

### How to resume
1. Wait for the first mismatch on H (minutes) or F2 (~12:30): `tapcheck.py` on all taps gives the
   first FIFO in stream order whose data is wrong and the cycle -> the node between the last good
   tap and the first bad one is the culprit (candidates after SWG_3: DWC_13, MVAU_3, DWC_14, Thr3,
   Mul, InnerShuffle, OuterShuffle; if FIFO_27/28 are already wrong: FMPadding_rtl_3 or SWG_3).
   Then correlate the cycle with the handshakes of that node (add a tap on its other side, or a
   small `rtlsim_trace` window around that cycle) and write a standalone testbench reproducing the
   handshake pattern. If G shows the same first mismatch as F2, iterate with G (faster).
2. The tap is an uncommitted diagnostic edit of `finn-rtllib/fifo/hdl/fifo_template.v` in
   `build/wt/fix007` (`git diff`); **revert it before any commit**. `subsim.py` (updated copy in
   `scripts/`) got `SUBSIM_TAPS=<FIFO,...>` (regenerates those FIFO wrappers with the tap) and a
   per-node `SUBSIM_BEHAV_MVU=<name,...>`; `SUBSIM_MODEL` selects another build's stitched model
   (the fold-target build reuses the same `ctx_hw.npz`).
3. Deletions this session (own scratch only): 36 finished stitch/rtlsim/codegen dirs of today's
   sub-sims in `build/finn_build/fx/fix007` (mtime 2026-10-20 before 11:44); nothing outside
   `build/wt/fix007`, `fx/fix007` and the scratchpad was deleted or written. Session 1's projects in
   `fx/fix007` (2026-10-19) and the three running sims' dirs are still there: delete when done.
4. OuterShuffle II=3 (981k cycles/frame, estimate 327k) is still the real throughput bug of #1725.

## Session 3 (2026-10-20, ~12:35-13:00): root cause found — MVU output queue overflow (dev RTL)

### Localization
`H2_foldtarget_head_taps` (fold-target design, taps FIFO_0..FIFO_17, run to 95k cycles, then stopped)
with `tapcheck.py` (now X-tolerant; skips a FIFO without a compute node in front):
* FIFO_1, FIFO_2, FIFO_3 (FMPadding_0, SWG_0, DWC_0): 0 mismatches, no X. FIFO_3 = MVAU_rtl_0's input.
* **FIFO_4 = MVAU_rtl_0's output: first X at word 28707, cycle 29791** (pixel 897 = row 7 col 1,
  neuron folds 3 and 4); 80 X words of 81206, all other words equal the Python reference (no word
  lost/added). X words come in pairs of consecutive folds (3/4 at column 1 of rows 7, 8, 9, ...;
  later 10/11 elsewhere).
* FIFO_5..FIFO_17: X inherited (FIFO_5 cycle 34428, FIFO_7 36025, ..., FIFO_17 36281 = the first X
  found in `H_foldtarget_full_taps`).
* `F2_full_taps` (committed folding): X from FIFO_20 at cycle 187672; `G_full_behavhead_taps`
  (behavioural MVAU_0/1): X from FIFO_23 at 188492. Same origin; behavioural cores do not help.
* MVAU_rtl_0: MW 75, MH 64, SIMD 75, PE 2 -> **SF = MW/SIMD = 1**, NF 32; INT8 x UINT8, DSP48E2,
  `mvu.sv` core. Committed folding and fold-target: SIMD 75; fold-max (passes): SIMD 25 (SF 3).
* Recorded handshakes before the first X: Thr0 pops FIFO_4 (depth 2) 1/cycle up to 29706, stalls
  29707-29717, pops 6, stalls 29724-29784 (FIFO_5 full), pops 29785-29791 (X at 29791), stall
  4 cycles, X at 29796. MVAU_0 takes one input per 32 cycles (never input-starved here).

### Mechanism
`finn-rtllib/mvu/mvu_vvu_axi.sv`, `blkOutput`: SRL output queue `OBuf[0:MAX_IN_FLIGHT]` with
`MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH = 3 + $clog2(SIMD+1) + (SIMD == 1)` (DSP48). On backpressure
the registered `OLock` stops feeding the core one cycle after the first stalled cycle, so up to
L + 1 results (L = actual core latency) are pushed without a pop; the queue needs L + 2 entries.
* `mvu.sv` latency is `3 + $clog2(SIMD) + (SIMD == 1) + 1` since `e1561a40f` (2025-07-29,
  T. Preußer, "Pipeline all levels with additive reductions"); the wrapper's copy of the formula was
  not updated (still so on `upstream/dev` 11ebf1abb). SIMD 75: L = 11, wrapper assumes 10.
* Independently, the queue is one entry short even when the formula matches (power-of-two SIMD).
* SIMD 75: 11 entries, needs 13, `OPtr` reaches 12. Overflow shifts the oldest results out and
  `OBuf[OPtr]` is read out of range -> X (sim); the assertion `Overflowing output queue` fires but
  FINN's XSI rtlsim does not show `$error`. Only SF = 1 can produce a result every cycle; SF >= 2 is
  safe. Hardware effect of the out-of-range SRL read: undefined (may be right by accident).

### Reproducer (standalone xsim, no FINN): `scripts/mvu_queue_tb.sv`, `scripts/mvu_queue_run.sh`
`mvu_vvu_axi` with MVAU_0's parameters (overridable `TB_MW`, `TB_SIMD`, `TB_VER`, `TB_STRUCT`),
inputs and weights always valid, reference dot product per output; tready random (`+READY_PCT`),
bursty (`+BURST`: stalls of 1..12 cycles) or replayed (`+REPLAY=<file>`: sink behind a 2-deep FIFO
popping at the recorded FIFO_4 cycles, `scripts/fifo4_pops_foldtarget.txt.gz`). 45-50k cycles each,
`xelab --mt 4`. Columns: outputs / X words / wrong non-X words / `OPtr` max / queue entries /
assertion count.

| run | RTL | config | tready | out | X | wrong | OPtr max / entries |
|---|---|---|---|---|---|---|---|
| orig50 | dev | SIMD 75 behav | 50 % | 17868 | 8 | 0 | 12 / 11 |
| origstruct | dev | SIMD 75 DSP48E2 primitives | 50 % | 17927 | 7 | 0 | 12 / 11 |
| replay_orig | dev | SIMD 75 | recorded | 37044 | 6 (first at output 27928; full design 28708) | 0 | 12 / 11 |
| origburst | dev | SIMD 75 | bursty | 10682 | 508 | 0 | 12 / 11 |
| simd96/64/32/3 burst | dev | SF 1 | bursty | ~10.6k | 479 / 207 / 244 / 1328 | 0 | 12/11, 11/11, 10/10, 7/6 |
| orig_simd64 | dev | SIMD 64 | 50 % | 17853 | 0 | 0 | 10 / 11 |
| v1 / v3 burst | dev | SIMD 75 DSP48E1 / DSP58 | bursty | 9575 / 9625 | 455 / 472 | 0 | 12/11, 7/7 |
| v3_simd24 | dev | DSP58 SIMD 24 | bursty | 7721 | 0 | 7712 | 3 / 4 (2017 assertions; TB reference vs DSP58 core unresolved) |
| orig_sf2 / sf2 burst | dev | MW 150 SIMD 75 | 50 % / bursty | | 0 / 0 | 0 | 5 / 11, 6 / 11 |
| orig_sf5 | dev | MW 75 SIMD 15 | 50 % | 9784 | 0 | 0 | 1 / 8 |
| patched50 / burst / replay | latency formula fixed | SIMD 75 | | | 1 / 165 / 2 | 0 | 12 / 12 |
| fixed_* (9 runs) | formula fixed + `MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1` | SIMD 3/64/75/96, behav + struct | 50 % / bursty / recorded | | **0** | 0 | L+1 / L+2 |

Patched files were scratch copies (`scratchpad/repro_mvu/mvu_vvu_axi_{patched,fixed}.sv`, the
diff is the two lines above); dev's RTL is unchanged. dev's `mvu/tb/mvu_axi_tb.sv` has no SF = 1
configuration.

### Where the fix belongs
dev (Xilinx/finn `finn-rtllib/mvu/mvu_vvu_axi.sv`), not our branches and not #1725: our branches
only choose foldings/FIFO depths that expose it (SF = 1 + burst/stall backpressure). Options and
recommendation: `ANALYSIS.md` section 7 (A: two-line dev fix + test, B: upstream issue, C: avoid
SF = 1 in the flow, D: OuterShuffle II = 3 separately).

### Housekeeping
* Stopped (own recorded PIDs, localization complete): H2 3232776, H 1769207, F2 1656574,
  G 1769285. Tap logs stay in the scratchpad `s/`.
* Deletion of the issue's scratch projects `build/finn_build/fx/fix007/*` (102 dirs, 699 MB: all
  stitch/rtlsim/FIFO-regen dirs of sessions 1-3) was **refused by the permission classifier**; nothing
  was deleted this session. To free the space: `rm -rf /home/lstasytis/finn/build/finn_build/fx/fix007/*`.
* The diagnostic tap edit of `finn-rtllib/fifo/hdl/fifo_template.v` in `build/wt/fix007` stays
  uncommitted (revert with `git -C build/wt/fix007 checkout finn-rtllib/fifo/hdl/fifo_template.v`
  when the worktree is no longer needed).
