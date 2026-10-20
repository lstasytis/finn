# Issue 011: transformer_radioml / vision wrong output and hang
(Saved by the orchestrator from the agent's hand-back; subagents cannot write report files.)

## Summary
1. Wrong output: two upstream (dev) bugs, both fixed (Python only); stitched-IP output now matches the golden
   (radioml fold-target max |diff| 0; vision fold-max SUCCESS, max |diff| 0.066, atol 0.5).
   * (a) `GlobalAccPool_hls` (radioml): `AccPool_Batch<ImgDim=get_normal_input_shape()[1]>` sums ImgDim x ImgDim
     pixels; the transformer mean pool `[1,1,64]` has ImgDim=1, so it summed one token and (self-restarting HLS block)
     emitted one result per token: near-zero logits, identical output for every folding, bogus 2.67x rtlsim fps.
     Fix `d98fb4183` (fix/011) = `d87bf52e3` (fix/011-up): non-square inputs get hlslib's accumulation loop inline over
     the real pixel count, square inputs keep the hlslib call; new 1x8 test case.
   * (b) `Requant_rtl` declares `idat` as `logic signed` (requant.sv / requant_axi.sv) but `_requant_rtl_possible`
     (specialize_layers.py, upstream e5c559b2b) accepts any integer input: the UINT8->INT8 requant (radioml, vision)
     turned every value >= 128 negative (node-by-node: Requant_rtl_1 is the first diverging node, 851/6144 wrong;
     192 -> -32 instead of 96). Fix `ffa6c0a5a` (fix/011) = `aff7232c9` (fix/011-up): `_requant_rtl_possible` returns
     `idt.signed()`, unsigned inputs get Requant_hls; new `test_requant_rtl_signed_input_only[INT8|UINT8]`.
   Reproducers on pure upstream: `scripts/repro_accpool.py`, `scripts/repro_requant.py`. Our branches not involved.
2. Hang (not fixed, FIFO branch): `heuristic_analytical` undersizes the MLP residual skip FIFO (radioml committed
   folding FIFO_90 = 20 words; vision fold-target 92 and 108) while the main branch must read a whole token (48 words,
   DWC 8->384 bit) before emitting -> deadlock. Only that FIFO at 128: radioml completes in 42012 cycles (unmodified:
   no output in 1.71M); vision skip FIFOs at 512: completes in 404756 cycles. Cause: the MVAU tree-model TAVs put the
   first output 44-953 cycles before the first input and the fork is treated as transparent
   (`scripts/tav_diag.py`, `scripts/fifo_override.py`). Suggested: a floor on fork->join skip edges = tokens the other
   branch must read before its first output, or causal MVAU profiles. Owner: feature/dev/analytical-fifo-sizing.
3. Issue 007's MVU queue overflow is not involved (fold builds have no SF=1 MVAU; the fixed mvu_vvu_axi.sv leaves the
   committed-folding output unchanged).
4. Still failing: radioml fold-target/max verify but rtlsim 3255 vs estimate 16276 fps (0.2x, like issue 010; the
   earlier 2.67x was bug (a)); `heuristic_rtlsim` fails in per-node characterization (a node times out, probably the
   two-input ElementwiseAdd_hls whose harness feeds one input only). Not root-caused.
5. Node-by-node verification itself aborts on upstream: `Requant_rtl.execute_node` asserts on the 3D [1,64,96] shape.
   Upstream's own config cannot build radioml: default "characterize" FIFO sizing refuses reconvergent paths.

## Branches
fix/011 = dev-finnexamples-features + d98fb4183 + ffa6c0a5a (worktree build/wt/fix011); fix/011-up = the same two
fixes on upstream/update/finn-examples (d87bf52e3, aff7232c9; worktree fix011-up) for an upstream PR.

## Tests
GAP unit tests (cppsim + rtlsim, 7x7 and 1x8) 8 passed; GAP timing-model tests 7 passed; GAP cppsim on fix/011-up 12
passed; test_requant_rtl_signed_input_only 2+2 passed; radioml fold-target-full / fold-max-full output SUCCESS (throughput
FAILED 0.2x); vision fold-max-full output SUCCESS; radioml heuristic_analytical-full deadlock; heuristic_rtlsim-full fails.

## Deletions (own scratch)
fx/fix011 sub-dirs and code_gen/rtlsim/tav/vivado (8.7 GB -> 470 MB), fx/fix011-up/unit, fx/fix011-up0 except the
baseline log dir, worktree build/wt/fix011-base. Stopped own PIDs 4175988, 1293199, 1301666, 2742815.
