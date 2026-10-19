# Report 007: espcn (KV260_SOM) stitched-IP rtlsim output differs from the golden — root cause NOT found
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
