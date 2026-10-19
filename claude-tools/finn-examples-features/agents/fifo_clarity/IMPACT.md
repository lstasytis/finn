# Impact of each FIFO sizing rule (Vivado 2024.2, ZCU104, stitched-IP rtlsim, 1 frame)

Committed in docs/finn/fifo_sizing_heuristics.rst, section "Impact of each rule" (branch feature/dev/analytical-fifo-sizing-clarity). Models never changed by any rule: tfc-w1a1, tfc-w2a2, cybersecurity-mlp (minimum depth everywhere). Throughput not measurable: mobilenet_v1 (stitched-IP rtlsim gives X output bits on ZCU104, 2024.2 and 2023.1), resnet50 (V80, no rtlsim here). 'throttleable consumers' is the same ablation as the throttled cap and is not listed separately.

Method: full sizer vs the sizer with one rule removed (ablation), on characterized fixtures; throughput from rtlsim of the ablated depths applied with auto_fifo_depths=False.

## Classification and reading of the numbers (same text as the docs section)

Two classes of rules:

* **depth-minimizing** rules take depth away from the peak occupancy; removing
  one makes FIFOs deeper and can never cost throughput. They are: the paced
  graph input, the slack relaxation (with its two helpers, the reset past the
  pacer and the pacer tolerance), the throttled cap, the two-frame split and
  the fill-frame-only evaluation of join/branch edges.
* **throughput-protecting** rules put depth back where a minimizing rule would
  over-minimize; removing one makes FIFOs shallower and risks a stall. The
  burst floor corrects the slack relaxation and the paced input (a bursty
  consumer needs its burst buffered whatever the average rate says); the cap
  margin corrects the throttled cap; the join/branch rule and the idle-window
  floor correct the slack relaxation and the cap on edges whose deadline is a
  join; the max of the causal and non-causal pass corrects each pass by the
  other; the measured global period corrects the caller's period estimate.

What the numbers say:

* **Heavy-duty depth minimizers**: the paced graph input (x2-x4 depth on every
  convolutional model without it, +970 % storage on mobilenet_v1), the slack
  relaxation together with its pacer reset and tolerance (x6 depth on
  mobilenet_v1, +25-40 % on cnv-w2a2) and the throttled cap (x6 depth on
  mobilenet_v1; it is the only model with a throttleable consumer whose burst
  exceeds 256). The two-frame split is the fourth: +40-115 % depth on the
  convolutional models, +89 % on resnet50. None of these costs throughput
  where it was simulated (cnv-w1a1, cnv-w2a2, gtsrb, kws, vgg10-radioml).
* **The one heavy-duty throughput protector** is the burst floor: it is what
  keeps a bursty consumer (an MVAU reading ``SF`` words at the start of each
  output window, a sliding window generator reading rows) fed once the slack
  relaxation and the paced input have taken the run-ahead away. Without it
  cnv-w2a2 loses 46 % and gtsrb 17 % of its throughput, cnv-w1a1 1.5 %; the
  depth it adds is 35-95 % of the total on those models, but only 13-24 % of
  the storage, because it lands on narrow edges. On kws the same rule
  over-sizes (4.3 kbit instead of 0.9 for no throughput), see `Burst floor`_.
* **Minor rules**: the causal/non-causal pass pair changes one token on one
  edge of the linear models and 2 % of resnet50's depth (non-causal larger);
  the cap margin changes 16 tokens on one resnet50 edge; the fill-frame-only
  rule 4 % of resnet50's depth; the idle-window floor 9 % of resnet50's
  storage (deep DWC -> FMPadding FIFOs inside residual branches). The join
  rule is minor in count (one model) but not in size: without it resnet50's
  shortcut FIFOs collapse (-80 % storage) into what the deadlock argument says
  cannot work; it was not simulated.
* **No effect on any model**: the measured global period (the builder already
  passes the slowest TAV period; the rule only matters for a caller that
  passes none) and, on the linear models, the cap margin. Both are
  candidates for removal if a resnet50 simulation shows the 16-token edge
  does not need the margin; the measured period is one line and guards
  ``DeriveFIFOSizes(period=None)``.

## Ranking (rule removed vs full sizer; depth/storage % = change of the total when the rule is removed)

| rule | class | models affected | median depth % | max depth % | median kbit % | max kbit % | worst fps % when removed | note |
|---|---|---|---|---|---|---|---|---|
| paced graph input | minimizing | 5 (cnv-w1a1, cnv-w2a2, gtsrb, mobilenet_v1, resnet50) | +117.5 | +213.3 | +189.9 | +969.5 | +0.0 (cnv-w1a1, 3 models simulated) |  |
| slack relaxation | minimizing | 5 (cnv-w2a2, gtsrb, vgg10-radioml, mobilenet_v1, resnet50) | +9.0 | +587.4 | +11.5 | +460.1 | +0.0 (cnv-w2a2, 3 models simulated) |  |
| slack reset past the pacer | minimizing | 2 (vgg10-radioml, mobilenet_v1) | +294.3 | +584.8 | +204.9 | +398.4 | +0.0 (vgg10-radioml, 1 models simulated) | part of slack relaxation |
| pacer tolerance | minimizing | 2 (vgg10-radioml, mobilenet_v1) | +294.3 | +584.8 | +204.9 | +398.4 | +0.0 (vgg10-radioml, 1 models simulated) | part of slack relaxation / cap |
| throttled cap | minimizing | 1 (mobilenet_v1) | +582.5 | +582.5 | +371.6 | +371.6 | - |  |
| two-frame split | minimizing | 6 (cnv-w1a1, cnv-w2a2, gtsrb, vgg10-radioml, mobilenet_v1, resnet50) | +47.6 | +113.6 | +35.6 | +87.8 | +0.0 (cnv-w1a1, 4 models simulated) |  |
| join/branch edges never relaxed | protecting | 1 (resnet50) | -85.0 | -85.0 | -79.6 | -79.6 | - | corrects slack relaxation, cap |
| burst floor | protecting | 7 (kws, cnv-w1a1, cnv-w2a2, gtsrb, vgg10-radioml, mobilenet_v1, resnet50) | -37.2 | -94.2 | -18.6 | -78.0 | -46.1 (cnv-w2a2, 5 models simulated) | corrects slack relaxation, paced input |
| idle-window floor (branch edges) | protecting | 1 (resnet50) | -8.9 | -8.9 | -9.7 | -9.7 | - | corrects the two-frame split |
| fill frame only on join/branch edges | minimizing | 1 (resnet50) | +4.2 | +4.2 | +5.3 | +5.3 | - | resnet50 only |
| non-causal pass (max of two passes) | protecting | 1 (resnet50) | -2.0 | -2.0 | -1.6 | -1.6 | - | corrects the causal pass |
| causal pass (max of two passes) | protecting | 4 (cnv-w1a1, gtsrb, mobilenet_v1, resnet50) | -0.0 | -0.4 | -0.0 | -0.1 | +0.0 (cnv-w1a1, 3 models simulated) | corrects the non-causal pass |
| cap margin | protecting | 1 (resnet50) | -0.0 | -0.0 | -0.0 | -0.0 | - | corrects the throttled cap |
| measured global period | protecting | 0 (none) | +0.0 | +0.0 | +0.0 | +0.0 | - | corrects the caller's estimate |

## Per model (rows: rule removed; depth %, storage %, rtlsim fps % relative to the full sizer)

### tfc-w1a1: full sizer depth 28, 18.3 kbit, - fps; reference (largefifo_rtlsim) - fps, - kbit

no rule changes a depth on this model

### tfc-w2a2: full sizer depth 28, 6.7 kbit, - fps; reference (largefifo_rtlsim) - fps, - kbit

no rule changes a depth on this model

### cybersecurity-mlp: full sizer depth 16, 3.7 kbit, - fps; reference (largefifo_rtlsim) - fps, - kbit

no rule changes a depth on this model

### kws: full sizer depth 452, 4.3 kbit, 25471 fps; reference (largefifo_rtlsim) 25471 fps, 0.9 kbit

| rule removed | depth | depth % | kbit | kbit % | fps | fps % |
|---|---|---|---|---|---|---|
| burst floor | 26 | -94.2 | 0.9 | -78.0 | 25471 | +0.0 |

### cnv-w1a1: full sizer depth 271, 11.9 kbit, 12255 fps; reference (largefifo_rtlsim) 12255 fps, 22.3 kbit

| rule removed | depth | depth % | kbit | kbit % | fps | fps % |
|---|---|---|---|---|---|---|
| paced graph input | 849 | +213.3 | 25.5 | +113.7 | 12255 | +0.0 |
| two-frame split | 416 | +53.5 | 18.4 | +54.2 | 12255 | +0.0 |
| burst floor | 176 | -35.1 | 9.7 | -18.6 | 12077 | -1.4 |
| causal pass (max of two passes) | 270 | -0.4 | 11.9 | -0.0 | 12255 | +0.0 |

### cnv-w2a2: full sizer depth 1405, 25.5 kbit, 3051 fps; reference (largefifo_rtlsim) 3045 fps, 83.5 kbit

| rule removed | depth | depth % | kbit | kbit % | fps | fps % |
|---|---|---|---|---|---|---|
| paced graph input | 4060 | +189.0 | 91.5 | +258.3 | 3051 | +0.0 |
| two-frame split | 3001 | +113.6 | 45.1 | +76.5 | 3051 | +0.0 |
| slack relaxation | 1974 | +40.5 | 32.0 | +25.1 | 3051 | +0.0 |
| burst floor | 882 | -37.2 | 22.2 | -13.1 | 1644 | -46.1 |
| causal pass (max of two passes) | 1405 | +0.0 | 25.5 | +0.0 | 3051 | +0.0 |

### gtsrb: full sizer depth 1298, 9.9 kbit, 8134 fps; reference (largefifo_rtlsim) 8134 fps, 26.7 kbit

| rule removed | depth | depth % | kbit | kbit % | fps | fps % |
|---|---|---|---|---|---|---|
| paced graph input | 2823 | +117.5 | 28.6 | +189.9 | 8134 | +0.0 |
| burst floor | 521 | -59.9 | 7.5 | -23.6 | 6751 | -17.0 |
| two-frame split | 1838 | +41.6 | 11.5 | +17.1 | 8134 | +0.0 |
| slack relaxation | 1415 | +9.0 | 10.8 | +9.3 | 8134 | +0.0 |
| causal pass (max of two passes) | 1297 | -0.1 | 9.8 | -0.1 | 8134 | +0.0 |

### vgg10-radioml: full sizer depth 131, 14.2 kbit, 243427 fps; reference (largefifo_rtlsim) 243427 fps, 14.0 kbit

| rule removed | depth | depth % | kbit | kbit % | fps | fps % |
|---|---|---|---|---|---|---|
| slack reset past the pacer | 136 | +3.8 | 15.8 | +11.5 | 243427 | +0.0 |
| pacer tolerance | 136 | +3.8 | 15.8 | +11.5 | 243427 | +0.0 |
| slack relaxation | 136 | +3.8 | 15.8 | +11.5 | 243427 | +0.0 |
| burst floor | 123 | -6.1 | 13.9 | -1.8 | 243427 | +0.0 |
| two-frame split | 135 | +3.1 | 14.4 | +1.7 | 243427 | +0.0 |

### mobilenet_v1: full sizer depth 9432, 124.1 kbit, - fps; reference (largefifo_rtlsim) - fps, - kbit

| rule removed | depth | depth % | kbit | kbit % | fps | fps % |
|---|---|---|---|---|---|---|
| paced graph input | 13281 | +40.8 | 1326.9 | +969.5 | - | - |
| slack relaxation | 64832 | +587.4 | 694.9 | +460.1 | - | - |
| slack reset past the pacer | 64587 | +584.8 | 618.3 | +398.4 | - | - |
| pacer tolerance | 64586 | +584.8 | 618.3 | +398.4 | - | - |
| throttled cap | 64369 | +582.5 | 585.1 | +371.6 | - | - |
| burst floor | 3003 | -68.2 | 43.4 | -65.0 | - | - |
| two-frame split | 9863 | +4.6 | 138.1 | +11.3 | - | - |

### resnet50: full sizer depth 362655, 2653.1 kbit, - fps; reference (largefifo_rtlsim) - fps, - kbit

| rule removed | depth | depth % | kbit | kbit % | fps | fps % |
|---|---|---|---|---|---|---|
| two-frame split | 686753 | +89.4 | 4983.8 | +87.8 | - | - |
| join/branch edges never relaxed | 54433 | -85.0 | 540.6 | -79.6 | - | - |
| paced graph input | 486488 | +34.1 | 4293.6 | +61.8 | - | - |
| idle-window floor (branch edges) | 330410 | -8.9 | 2394.7 | -9.7 | - | - |
| fill frame only on join/branch edges | 378030 | +4.2 | 2794.3 | +5.3 | - | - |
| burst floor | 358497 | -1.1 | 2597.4 | -2.1 | - | - |
| non-causal pass (max of two passes) | 355475 | -2.0 | 2610.9 | -1.6 | - | - |



## Raw numbers (impact.json)

```
{
 "cnv-w1a1": {
  "full": {
   "depth": 271,
   "kbit": 11.916015625,
   "fps": 12254.901960784313,
   "fps_sim_kbit": 11.916015625
  },
  "rules": {
   "no_burst_floor": {
    "depth": 176,
    "kbit": 9.705078125,
    "depth_pct": -35.055350553505534,
    "kbit_pct": -18.55433535485986,
    "fps": 12077.294685990339,
    "fps_pct": -1.4492753623188304
   },
   "no_cap": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 270,
    "kbit": 11.912109375,
    "depth_pct": -0.36900369003690037,
    "kbit_pct": -0.03278151122766759,
    "fps": 12254.901960784313,
    "fps_pct": 0.0
   },
   "no_fill_only": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 416,
    "kbit": 18.376953125,
    "depth_pct": 53.505535055350556,
    "kbit_pct": 54.2206195705622,
    "fps": 12254.901960784313,
    "fps_pct": 0.0
   },
   "no_idle_floor": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 849,
    "kbit": 25.462890625,
    "depth_pct": 213.28413284132841,
    "kbit_pct": 113.68628093755122,
    "fps": 12254.901960784313,
    "fps_pct": 0.0
   },
   "no_join_rule": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_throttleable": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 271,
    "kbit": 11.916015625,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": {
   "fps": 12254.901960784313,
   "kbit": 22.314453125
  }
 },
 "cnv-w2a2": {
  "full": {
   "depth": 1405,
   "kbit": 25.544921875,
   "fps": 3051.199121254653,
   "fps_sim_kbit": 25.544921875
  },
  "rules": {
   "no_burst_floor": {
    "depth": 882,
    "kbit": 22.203125,
    "depth_pct": -37.22419928825623,
    "kbit_pct": -13.082039911308204,
    "fps": 1644.2230224107598,
    "fps_pct": -46.11223466350976
   },
   "no_cap": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": 3051.199121254653,
    "fps_pct": 0.0
   },
   "no_fill_only": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 3001,
    "kbit": 45.078125,
    "depth_pct": 113.59430604982207,
    "kbit_pct": 76.46609067971558,
    "fps": 3051.199121254653,
    "fps_pct": 0.0
   },
   "no_idle_floor": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 4060,
    "kbit": 91.52734375,
    "depth_pct": 188.96797153024912,
    "kbit_pct": 258.2995641868644,
    "fps": 3051.199121254653,
    "fps_pct": 0.0
   },
   "no_join_rule": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 1974,
    "kbit": 31.966796875,
    "depth_pct": 40.49822064056939,
    "kbit_pct": 25.139536661824298,
    "fps": 3051.199121254653,
    "fps_pct": 0.0
   },
   "no_throttleable": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 1405,
    "kbit": 25.544921875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": {
   "fps": 3045.0669914738123,
   "kbit": 83.505859375
  }
 },
 "cybersecurity-mlp": {
  "full": {
   "depth": 16,
   "kbit": 3.6845703125,
   "fps": null,
   "fps_sim_kbit": null
  },
  "rules": {
   "no_burst_floor": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_fill_only": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_idle_floor": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_join_rule": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_throttleable": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 16,
    "kbit": 3.6845703125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": null
 },
 "gtsrb": {
  "full": {
   "depth": 1298,
   "kbit": 9.8564453125,
   "fps": 8134.049129656743,
   "fps_sim_kbit": 9.8564453125
  },
  "rules": {
   "no_burst_floor": {
    "depth": 521,
    "kbit": 7.533203125,
    "depth_pct": -59.8613251155624,
    "kbit_pct": -23.57079163776875,
    "fps": 6750.826976304597,
    "fps_pct": -17.00533315331128
   },
   "no_cap": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 1297,
    "kbit": 9.8486328125,
    "depth_pct": -0.07704160246533127,
    "kbit_pct": -0.07926285544436738,
    "fps": 8134.049129656743,
    "fps_pct": 0.0
   },
   "no_fill_only": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 1838,
    "kbit": 11.537109375,
    "depth_pct": 41.60246533127889,
    "kbit_pct": 17.051421777469532,
    "fps": 8134.049129656743,
    "fps_pct": 0.0
   },
   "no_idle_floor": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 2823,
    "kbit": 28.5732421875,
    "depth_pct": 117.4884437596302,
    "kbit_pct": 189.89398593084317,
    "fps": 8134.049129656743,
    "fps_pct": 0.0
   },
   "no_join_rule": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 1415,
    "kbit": 10.7705078125,
    "depth_pct": 9.01386748844376,
    "kbit_pct": 9.273754086990984,
    "fps": 8134.049129656743,
    "fps_pct": 0.0
   },
   "no_throttleable": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 1298,
    "kbit": 9.8564453125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": {
   "fps": 8134.049129656743,
   "kbit": 26.658203125
  }
 },
 "kws": {
  "full": {
   "depth": 452,
   "kbit": 4.26953125,
   "fps": 25471.21752419766,
   "fps_sim_kbit": 4.26953125
  },
  "rules": {
   "no_burst_floor": {
    "depth": 26,
    "kbit": 0.94140625,
    "depth_pct": -94.24778761061947,
    "kbit_pct": -77.95059469350412,
    "fps": 25471.21752419766,
    "fps_pct": 0.0
   },
   "no_cap": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_fill_only": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_idle_floor": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_join_rule": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_throttleable": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 452,
    "kbit": 4.26953125,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": {
   "fps": 25471.21752419766,
   "kbit": 0.85546875
  }
 },
 "mobilenet_v1": {
  "full": {
   "depth": 9432,
   "kbit": 124.0654296875,
   "fps": null,
   "fps_sim_kbit": null
  },
  "rules": {
   "no_burst_floor": {
    "depth": 3003,
    "kbit": 43.4482421875,
    "depth_pct": -68.1615776081425,
    "kbit_pct": -64.97957384507608,
    "fps": null,
    "fps_pct": null
   },
   "no_cap": {
    "depth": 64369,
    "kbit": 585.1435546875,
    "depth_pct": 582.4533502968618,
    "kbit_pct": 371.64109789598797,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 9432,
    "kbit": 124.0654296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 9431,
    "kbit": 124.0576171875,
    "depth_pct": -0.010602205258693808,
    "kbit_pct": -0.006297080516045749,
    "fps": null,
    "fps_pct": null
   },
   "no_fill_only": {
    "depth": 9432,
    "kbit": 124.0654296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 9863,
    "kbit": 138.1357421875,
    "depth_pct": 4.569550466497032,
    "kbit_pct": 11.341042009398393,
    "fps": null,
    "fps_pct": null
   },
   "no_idle_floor": {
    "depth": 9432,
    "kbit": 124.0654296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 13281,
    "kbit": 1326.8779296875,
    "depth_pct": 40.80788804071247,
    "kbit_pct": 969.4985162504034,
    "fps": null,
    "fps_pct": null
   },
   "no_join_rule": {
    "depth": 9432,
    "kbit": 124.0654296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 9432,
    "kbit": 124.0654296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 64587,
    "kbit": 618.3154296875,
    "depth_pct": 584.7646310432569,
    "kbit_pct": 398.3785017671182,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 64586,
    "kbit": 618.3076171875,
    "depth_pct": 584.7540288379982,
    "kbit_pct": 398.3722046866022,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 64832,
    "kbit": 694.8779296875,
    "depth_pct": 587.362171331637,
    "kbit_pct": 460.08989082436653,
    "fps": null,
    "fps_pct": null
   },
   "no_throttleable": {
    "depth": 64369,
    "kbit": 585.1435546875,
    "depth_pct": 582.4533502968618,
    "kbit_pct": 371.64109789598797,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 9432,
    "kbit": 124.0654296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": null
 },
 "resnet50": {
  "full": {
   "depth": 362655,
   "kbit": 2653.1201171875,
   "fps": null,
   "fps_sim_kbit": null
  },
  "rules": {
   "no_burst_floor": {
    "depth": 358497,
    "kbit": 2597.3779296875,
    "depth_pct": -1.1465442362576002,
    "kbit_pct": -2.1010050445469752,
    "fps": null,
    "fps_pct": null
   },
   "no_cap": {
    "depth": 362655,
    "kbit": 2653.1201171875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 362639,
    "kbit": 2652.8701171875,
    "depth_pct": -0.004411906633025878,
    "kbit_pct": -0.00942286775409996,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 362654,
    "kbit": 2653.0888671875,
    "depth_pct": -0.0002757441645641174,
    "kbit_pct": -0.001177858469262495,
    "fps": null,
    "fps_pct": null
   },
   "no_fill_only": {
    "depth": 378030,
    "kbit": 2794.3232421875,
    "depth_pct": 4.239566530173305,
    "kbit_pct": 5.322153493362584,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 686753,
    "kbit": 4983.7568359375,
    "depth_pct": 89.36813224690133,
    "kbit_pct": 87.84512633452285,
    "fps": null,
    "fps_pct": null
   },
   "no_idle_floor": {
    "depth": 330410,
    "kbit": 2394.6904296875,
    "depth_pct": -8.891370586369966,
    "kbit_pct": -9.740595076183517,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 486488,
    "kbit": 4293.6357421875,
    "depth_pct": 34.146227130468354,
    "kbit_pct": 61.833447131638565,
    "fps": null,
    "fps_pct": null
   },
   "no_join_rule": {
    "depth": 54433,
    "kbit": 540.5732421875,
    "depth_pct": -84.99041789028139,
    "kbit_pct": -79.62499930984855,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 362655,
    "kbit": 2653.1201171875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 362655,
    "kbit": 2653.1201171875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 362655,
    "kbit": 2653.1201171875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 362666,
    "kbit": 2653.3779296875,
    "depth_pct": 0.0030331858102052916,
    "kbit_pct": 0.009717332371415583,
    "fps": null,
    "fps_pct": null
   },
   "no_throttleable": {
    "depth": 362655,
    "kbit": 2653.1201171875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 355475,
    "kbit": 2610.8974609375,
    "depth_pct": -1.979843101570363,
    "kbit_pct": -1.5914340242822884,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": null
 },
 "tfc-w1a1": {
  "full": {
   "depth": 28,
   "kbit": 18.279296875,
   "fps": null,
   "fps_sim_kbit": null
  },
  "rules": {
   "no_burst_floor": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_fill_only": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_idle_floor": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_join_rule": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_throttleable": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 28,
    "kbit": 18.279296875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": null
 },
 "tfc-w2a2": {
  "full": {
   "depth": 28,
   "kbit": 6.6875,
   "fps": null,
   "fps_sim_kbit": null
  },
  "rules": {
   "no_burst_floor": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_fill_only": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_idle_floor": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_join_rule": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_tol": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_slack": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_throttleable": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 28,
    "kbit": 6.6875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": null
 },
 "vgg10-radioml": {
  "full": {
   "depth": 131,
   "kbit": 14.1796875,
   "fps": 243427.45861733204,
   "fps_sim_kbit": 14.1796875
  },
  "rules": {
   "no_burst_floor": {
    "depth": 123,
    "kbit": 13.9296875,
    "depth_pct": -6.106870229007634,
    "kbit_pct": -1.7630853994490359,
    "fps": 243427.45861733204,
    "fps_pct": 0.0
   },
   "no_cap": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_cap_margin": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_causal_pass": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_fill_only": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_frame_split": {
    "depth": 135,
    "kbit": 14.4140625,
    "depth_pct": 3.053435114503817,
    "kbit_pct": 1.6528925619834711,
    "fps": 243427.45861733204,
    "fps_pct": 0.0
   },
   "no_idle_floor": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_input_pacing": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_join_rule": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_measured_period": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "no_pacer_reset": {
    "depth": 136,
    "kbit": 15.8046875,
    "depth_pct": 3.816793893129771,
    "kbit_pct": 11.460055096418733,
    "fps": 243427.45861733204,
    "fps_pct": 0.0
   },
   "no_pacer_tol": {
    "depth": 136,
    "kbit": 15.8046875,
    "depth_pct": 3.816793893129771,
    "kbit_pct": 11.460055096418733,
    "fps": 243427.45861733204,
    "fps_pct": 0.0
   },
   "no_slack": {
    "depth": 136,
    "kbit": 15.8046875,
    "depth_pct": 3.816793893129771,
    "kbit_pct": 11.460055096418733,
    "fps": 243427.45861733204,
    "fps_pct": 0.0
   },
   "no_throttleable": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   },
   "only_causal_pass": {
    "depth": 131,
    "kbit": 14.1796875,
    "depth_pct": 0.0,
    "kbit_pct": 0.0,
    "fps": null,
    "fps_pct": null
   }
  },
  "reference": {
   "fps": 243427.45861733204,
   "kbit": 13.953125
  }
 }
}
```