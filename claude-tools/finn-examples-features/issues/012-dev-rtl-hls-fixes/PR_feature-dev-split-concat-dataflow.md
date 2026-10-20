# StreamingSplit_hls / StreamingConcat_hls: one word per cycle (dataflow top)

Branch `feature/dev/split-concat-dataflow` (one commit on `upstream/dev` 11ebf1abb).

## Problem

`StreamingSplit_hls` and `StreamingConcat_hls` process one folded word every **five** cycles, but
`get_exp_cycles()` (one cycle per folded word) and every other part of FINN assume one word per
cycle. In a design where a Split/Concat sits on the critical path the real throughput is up to 5x
below the estimate, and the folding optimizer never sees these nodes as the bottleneck.

Observed on finn-examples `transformer_benchmark` (ZCU104, maximize folding): estimate 2048 cycles
per frame, stitched-IP rtlsim 10240 (0.2x). Per-node rtlsim: `StreamingSplit_hls` 10240 cycles for
2048 words; all other node types within their estimates. All 15 Split/Concat IPs of the
finn-examples builds (2-4 streams, 4-22 bit) report the same csynth Interval 5.

## Root cause

The generated HLS top calls the finn-hlslib function once per invocation:

```cpp
void StreamingSplit_hls_0(hls::stream<hls::vector<ap_int<8>, 2>> &in0_V, /* out0_V..out3_V */)
{
#pragma HLS INTERFACE axis port=in0_V
...
#pragma HLS INTERFACE ap_ctrl_none port=return
...
StreamingSplit<4, 4, 4, 4>(in0_V, out0_V, out1_V, out2_V, out3_V);
}
```

`StreamingSplit` / `StreamingConcat` in finn-hlslib (`split.hpp`, `concat.hpp`) are single-word,
free-running functions (`#pragma HLS pipeline II=1 style=flp`, static selector/counter state, one
`if(!src.empty())` word per call). Without a dataflow region Vitis HLS schedules the top as an
unpipelined function around this call: csynth of the top reports **Latency 4, Interval 5,
Pipeline no**, so each word costs a full top-level invocation. The inner function itself is II=1.

`DuplicateStreams_hls` wraps the same kind of hlslib function (`StreamingDup` in `dup.hpp`, also
`pipeline II=1 style=flp`) and adds `#pragma HLS dataflow disable_start_propagation` to its top;
it runs at one word per cycle.

## Change

Add `#pragma HLS dataflow disable_start_propagation` to the generated top of both ops
(`split_hls.py`, `concat_hls.py`, `pragmas()`), exactly as `duplicatestreams_hls.py` does. No
change to `get_exp_cycles()`: the existing one-cycle-per-word estimate becomes correct.

csynth of the top (xc7z020, 10 ns, the 6 test configurations):

| | Latency | Interval | Pipeline | FF | LUT |
|---|---|---|---|---|---|
| before | 4 | 5 | no | 39-177 | 198-264 |
| after | 2 | **1** | dataflow | 26-53 | 115-143 |

## Tests

New `test_fpgadataflow_split_exp_cycles_rtlsim[simd 1,2,4]` and
`test_fpgadataflow_concat_exp_cycles_rtlsim[simd 1,2,4]`: one Split (4 x 8 channels) or Concat
(4 x 8 channels) node, stitched IP, steady-state rtlsim cycles per frame (`xsi_fifosim`, 4 frames)
must equal `get_exp_cycles()` = folded words (128 / SIMD) within 2 %.

* With this change: 6 passed.
* Without the pragma (plain `upstream/dev`): 6 failed, measured exactly 5x (e.g. 640 vs 128 at
  SIMD 1, 320 vs 64 at SIMD 2, 160 vs 32 at SIMD 4).
* Existing `test_fpgadataflow_split.py` / `test_fpgadataflow_concat.py` (cppsim, node rtlsim,
  stitched-IP rtlsim, values checked): 6 passed on this branch.

## Relation to `feature/dev/hls-exp-cycles-overhead`

That branch contains the alternative fix (commit 143b54492, "StreamingSplit_hls,
StreamingConcat_hls: five cycles per word in get_exp_cycles"): it keeps the slow hardware and
makes the estimate honest (`get_exp_cycles()` x 5), with the same two rtlsim tests expecting 5x.
The two are mutually exclusive:

* This PR fixes the hardware (5x faster Split/Concat, fewer LUTs/FFs), so the x5 override would
  then over-estimate by 5x. If this PR is accepted, commit 143b54492 must be dropped from
  `feature/dev/hls-exp-cycles-overhead` (its MVAU_hls/LabelSelect_hls overhead commit is
  independent and stays).
* Merging both branches as they are conflicts in the two test files (5x vs 1x expectation), and
  the source files merge silently into "pragma + x5 estimate", which is wrong.
* The x5 estimate is the fallback if a dataflow region in these tops is not wanted.

## Risk

Low. The pragma is the one `DuplicateStreams_hls` has used with the same hlslib function style;
the C++ (cppsim) path ignores it. Behavioural difference: the top is now a free-running dataflow
process (as DuplicateStreams), i.e. it accepts a word whenever the input is valid and all outputs
can take it; there is no `ap_ctrl` interface either way (`ap_ctrl_none`). The latency drops from 4 to
2 cycles, which only shortens pipelines (FIFO sizing by simulation and the analytical models see
a faster node).
