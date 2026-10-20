# mvu_vvu_axi: size the output queue for the core's real pipeline depth

Branch `feature/dev/mvu-output-queue` (one commit on `upstream/dev` 11ebf1abb): RTL fix in
`finn-rtllib/mvu/mvu_vvu_axi.sv` plus a FINN test.

## Problem

An `MVAU_rtl` at **SF = 1** (SIMD = MW: one complete dot product, i.e. one output word, per cycle)
silently produces wrong outputs when its output stream sees bursty backpressure (full-rate
acceptance followed by a stall of at least the core latency). In simulation the lost words appear
as X; FINN's XSI rtlsim does not surface the RTL's own `$error` ("Overflowing output queue"), so
nothing flags it. On hardware the read is out of range of the SRL queue (undefined; it may be right
by accident).

Found on finn-examples `espcn-bsd300` (KV260_SOM): ~10 % of the output values were 0 in stitched-IP
rtlsim, in fixed column ranges of every row from a fixed row on. The Python HW model matched the
golden output; every non-zero rtlsim value was exact. Tapping every FIFO of the design and comparing
with the Python HW-model streams located the first wrong word at the output of `MVAU_rtl_0`
(MW 75, MH 64, **SIMD 75**, PE 2, INT8 x UINT8, DSP48E2), cycle 29791, right after the downstream
`Thresholding_rtl` (input FIFO depth 2) accepted 1 word/cycle and then stalled for 61 cycles
(its own output FIFO was full: the design's tail is slower). The X words replace values one for one;
downstream they became the zeros. Every input stream of `MVAU_rtl_0` was correct. The same design
with SIMD 25 (SF 3) passes.

## Root cause

`blkOutput` of `mvu_vvu_axi` catches every core result in an SRL queue `OBuf[0:MAX_IN_FLIGHT]`.
When the output stalls, the registered `OLock` stops feeding the core, and the queue must absorb
everything still in the core pipeline. It was too small in three ways:

1. **Stale latency formula.** Commit e1561a40f ("Pipeline all levels with additive reductions",
   2025-07-29) changed `mvu.sv` to `PIPELINE_DEPTH = 3 + $clog2(SIMD) + (SIMD == 1) + 1`; the
   wrapper kept `3 + $clog2(SIMD+1) + (SIMD == 1)`, one cycle short for every non-power-of-two
   SIMD (SIMD 75: core 11, wrapper 10).
2. **Off by one.** `OLock` is registered: the input accepted in the cycle in which the stall is
   detected still enters the core, so up to depth + 1 results arrive without a pop. The queue had
   `MAX_IN_FLIGHT + 1 = depth + 1` entries and needs depth + 2.
3. **Wrong core.** The depth was chosen by `VERSION == 3` (DSP58), but on DSP58 only configurations
   with `NUM_LANES <= 3`, W <= 8 and A <= 9 bit use `mvu_vvu_8sx9_dsp58`; all others (e.g. 4-bit x
   4-bit, where 4 lanes fit) use `mvu.sv`, whose pipeline is several cycles deeper (SIMD 24: 9 vs 4).

In addition, when `MAX_IN_FLIGHT + 1` is a power of two, an overflow wraps the signed `OPtr` to
negative ("empty"), dropping queued words entirely (this explains the unresolved "wrong values"
result of DSP58 SIMD 24 below: words lost, the reference queue misaligned).

With SF >= 2 at most every second pipeline slot carries a result and the queue never fills, which
is why only SF = 1 is affected and dev's `mvu/tb/mvu_axi_tb.sv` (no SF = 1 configuration; its
output checker accepts at most every second cycle) never saw it.

## Change

```systemverilog
// module level (moved out of blkDsp, unchanged lane computation)
localparam bit  USE_8SX9_DSP58 = !IS_MVU || ((VERSION > 2) && (NUM_LANES <= 3) && (WEIGHT_WIDTH <= 8) && (ACTIVATION_WIDTH <= 9));
...
if(USE_8SX9_DSP58) begin : genINT8          // was the same condition inline
...
localparam int unsigned  CORE_PIPELINE_DEPTH =
	USE_8SX9_DSP58? 3 + (SEGMENTLEN == 0? 0 : ((SIMD+2)/3 -1)/SEGMENTLEN) :   // was VERSION == 3
	/* else */      3 + $clog2(SIMD) + (SIMD == 1) + 1;	// = PIPELINE_DEPTH of mvu.sv
localparam int unsigned  MAX_IN_FLIGHT = CORE_PIPELINE_DEPTH + 1;            // was + 0
```

The DSP58 INT8 formula already matched `mvu_vvu_8sx9_dsp58` (latency `MAX_PIPELINE_STAGES + 2` =
`3 + (CHAINLEN-1)/SEGLEN`); only the off-by-one applies there. Cost: 1-2 SRL entries per output
queue (DSP48, DSP58 INT8; `PE * ACCU_WIDTH` bits wide), the `mvu.sv` depth for the generic DSP58
core. Nothing changes while the queue does not overflow.

## Tests

### FINN test (in this PR)

`tests/fpgadataflow/test_fpgadataflow_mvau.py::test_fpgadataflow_rtl_mvau_output_backpressure`
`[part x idt_wdt]`: `MVAU_rtl` (MW = SIMD = 24, MH 32, PE 1: SF 1, NF 32, non-power-of-two SIMD)
feeding an `MVAU_hls` (MW 32, MH 16, SIMD = PE = 1) through a depth-2 FIFO. The HLS MVAU reads its
input only in its first neuron fold: 32 words at one per cycle, then 15 x 32 = 480 cycles of stall
while it replays its buffer; that is the burst-then-stall pattern. 64 input vectors, stitched-IP
rtlsim, exact comparison with numpy. Parts: xczu7ev (DSP48E2) and xcvc1902 (DSP58); UINT4 x INT4 and
UINT8 x INT8 (on DSP58: generic `mvu.sv` core and INT8 core). Values are kept exact in float32.

| RTL | zu7ev 4-bit | zu7ev 8-bit | vc1902 4-bit (generic core) | vc1902 8-bit (INT8 core) |
|---|---|---|---|---|
| upstream/dev | FAIL | FAIL | FAIL | FAIL |
| fixes 1 + 2 only | pass | pass | FAIL (rtlsim timeout: words lost) | pass |
| this PR (1 + 2 + 3) | **pass** | **pass** | **pass** | **pass** |

Regression on this branch, existing `test_fpgadataflow_rtl_mvau` (cppsim, node rtlsim and
stitched-IP rtlsim against numpy; PE 1/9/18 x SIMD 1/16/32 incl. SF = 1 without backpressure, tiled
and external-memory variants; UINT4 x INT4 and UINT8 x INT8), clk 1.66 ns, no pumped memory:
without pumped compute on xcvc1902 and xc7z020, with pumped compute on xcvc1902: **72 passed**, 36
skipped by the test itself. (The xcku3p cases cannot run here: "Specified part could not be found"
in this Vivado install, identical on `upstream/dev`.)

### Standalone reproducer (xsim, not in this PR)

`mvu_vvu_axi` alone with `MVAU_rtl_0`'s parameters, inputs and weights always valid, every output
checked against a dot-product reference; output tready 50 % random, bursty (stalls of 1-12 cycles)
or replaying the tready pattern recorded in the espcn design. 45-50 k cycles each.

| RTL | configurations | X / wrong words | `OPtr` max / entries |
|---|---|---|---|
| dev | SIMD 75 DSP48E2 (behav and structural), recorded pattern | 6-508 X | 12 / 11 |
| dev | SF 1, SIMD 96 / 64 / 32 / 3, DSP48E1, DSP58 | 207-1328 X | over by 1-2 |
| dev | DSP58 INT8, SIMD 24 | 0 X, 4992 of 4994 wrong (OPtr wrapped) | 3 / 4 |
| dev | DSP58 generic core (4 x 4 bit), SIMD 32 | 999 X, 3512 wrong | 7 / 5 |
| dev | SF 2, SF 5 | 0 | 6 / 11, 1 / 8 |
| this PR | SIMD 1, 3, 24 (DSP58), 32 (DSP58 4-bit, behav + struct), 64, 75 (DSP48E1/E2, DSP58, behav + struct, 4- and 8-bit), 96 (DSP48E1 4-bit), SF 2, recorded pattern | **0 / 0**, no assertion | L+1 / L+2 (tight) |

### End to end (espcn)

The two failing espcn designs, re-stitched unchanged (same IPs, foldings and FIFO depths) with
this RTL, full-frame stitched-IP rtlsim:

| design | before (dev RTL) | with this PR |
|---|---|---|
| committed folding, analytic FIFOs (MVAU_rtl_0 SIMD 75 = MW) | 20460 zero outputs | 0 wrong vs Python HW model; golden within 1/255 (12 known rounding spots); 999,901 cycles |
| optimizer folding at the committed throughput | 31920 zero outputs | 0 wrong; golden within 1/255; 1,200,526 cycles |

Both outputs are bit-identical to the output of the espcn build that already passed (SF = 3 for
MVAU_rtl_0).

## Risk

Low. Only queue sizing changes; the queue's logic is unchanged and an SRL entry costs one LUT per
bit at most (SRL16/32 already in place). Designs that never overflowed behave identically. The
moved localparams (`MIN_LANE_WIDTH`, `A_WIDTH`, `NUM_LANES`) are now module-level; no other file
references them (`mvu_tiled_axi.sv` has its own). With `PUMPED_COMPUTE` the core runs at 2x on
`DSP_SIMD = SIMD/2`; the formula uses SIMD and stays conservative there (not exercised by the new
test).
