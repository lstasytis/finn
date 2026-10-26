# 024 Part B: resnet50 full and transformer_radioml / vision wrong outputs

Verdict: **neither is FIFO-related.**
- resnet50 full: a stream-layout bug in the committed V80 folding (depthwise SWG SIMD 2048 -> DWC -> Pool PE 4).
- transformer_radioml / vision: the runs resumed from stale cached reference checkpoints that still contain the issue-011 bug (`Requant_rtl` on a UINT8 input). The code fix is already landed; the cached checkpoints were converted before it.

## Item 3: resnet50 full (V80), heuristic_analytical, wrong output

Build `fx/fifo2/fifo-heuristic_analytical_resnet50_V80_rf8r_hmr`: rtlsim runs at 0.9998x the estimate. Its top-5 output is `[714 315 683 227 571]`; the golden is `[0 390 48 391 150]` (no label in common).

**Cause (confirmed):**
- `resnet50_folding_config_V80.json` (upstream PR #1725, commit 13081fe57) sets `Pool_hls_1` (global average pool, QuantAvgPool 7x7, 2048 channels) to PE 4.
- It has no entry for that pool's depthwise SWG `ConvolutionInputGenerator_rtl_21`. The SWG therefore keeps InferPool's default SIMD = IFMChannels = 2048.
- InsertDWC then puts a 12288 -> 24 bit DWC between the two nodes without reordering:
  - The depthwise RTL SWG emits `(C/SIMD, K, SIMD)` = `(1, 49, 2048)`, i.e. kernel-major.
  - Pool_hls (hlslib Pool_batch) reads `(C/PE, K, PE)` = `(512, 49, 4)`.
  - So Pool output group g = the sum of stream words 49g..49g+48, which mixes kernel positions and channels.
- Consistent with this:
  - The small folding (SWG SIMD 1 = Pool PE 1, no DWC) passes.
  - `Pool_hls_0` (max pool, SWG SIMD 4 = PE 4) is fine.
  - The islands notes record the same bug for the U55C folding_pe_div2.

**Node-level check** (`scripts/r50_pool_node_check.py <final model> <part> asbuilt|fixed`):
- The script takes the 3 nodes from the failing build's final model and drops the FIFOs.
- It runs per-node rtlsim in its own FINN_BUILD_DIR on a random INT6 1x7x7x2048 input.
- Each variant runs in its own process, because finn_xsi caches the loaded simulation per top-module name.

| variant | vs clip(floor(channel sum / 64)) (correct) | vs kernel-major mis-grouped sums / 64 |
|---|---|---|
| as built: SWG SIMD 2048 -> DWC -> Pool PE 4 | 1772 of 2048 differ | **0 of 2048 differ** |
| fixed: SWG SIMD 4 = PE, no DWC | **0 of 2048 differ** | 1772 of 2048 differ |

The wrong sums are deterministic and fixed by the folding's stream order; FIFO depth and backpressure play no role.

Side note: FINN's Python Pool reference shifts by 8 for INT6 -> INT4, the HLS core by 6. The golden comes from the QONNX model, so this is irrelevant here.

**Fix (not committed; for the PR / #1725 owner):**
- Add `"ConvolutionInputGenerator_rtl_21": {"SIMD": 4}` to `resnet50_folding_config_V80.json`.
- The DWC and the 12288-bit streams disappear.
- The SWG then takes 49*2048/4 = 25088 cycles, far below the ~3.6 M-cycle bottleneck, so the estimate is unchanged.
- resnet50's `est_*` caches must be moved aside afterwards.

General alternative: InsertDWC/SetFolding could force a depthwise SWG's SIMD to its Pool consumer's PE when a folding JSON leaves it unset. The optimizer already does this (`set_folding.py` lists Pool_hls in `depthwise_op_exceptions`), so optimizer foldings are not affected. The FIFO sizing PR is not involved.

## Item 4: transformer_radioml / vision small, wrong output (issue 019)

**Cause: stale reference checkpoints.** 019's `stale_requant_scan.py` finds `Requant_rtl_1(UINT8)` in every radioml/vision checkpoint the failing runs resumed from:
- `fx/fifo2/est_transformer_small_transformer_radioml_ZCU104_d10drro0`
- `fx/fifo2/est_transformer_small_transformer_vision_ZCU104__644qarm`
- `fx/fifo2/est_transformer_transformer_radioml_ZCU104_c2e0_l7c`
- `fx/fifo2/est_transformer_transformer_vision_ZCU104_zcbs21n9`
- in folding2 and dwc2, the same stale dirs, now moved aside in `superseded_021/`.

How the stale checkpoints arose:
- `Requant_rtl` declares its input `logic signed` (issue 011 (b)), so every UINT8 value >= 128 turns negative.
- The fix "SpecializeLayers: Requant_rtl only for signed integer inputs" (34de4763e, 2026-10-20 16:21) is on dev-finnexamples-features and on all test2/* branches.
- The checkpoints were converted earlier, 2026-10-20 11:49-13:06.
- `cached_build` matches `est_*` by name prefix only. Every later FIFO, folding and DWC run therefore reused the pre-fix conversion.

Not affected:
- transformer_benchmark and transformer_language have no unsigned Requant.
- The islands2 checkpoints are newer and clean.

On 019's planned bisection:
- It is moot: the split-concat-dataflow and mvu-output-queue merges are not involved. The regression only coincided with them because the checkpoints were regenerated at about the same time, with pre-fix code.
- 019's previous agent had found this cause (fix/019-fifo cb8f2ae67, fingerprint-keyed caches) but never verified it or reported.

**FIFO depths not involved:**
- The heuristic_analytical and heuristic_rtlsim builds give **bitwise identical** wrong outputs although their FIFO depths differ:
  - radioml: 31 of 92 FIFOs differ (total depth 46630 vs 50700);
  - vision: 43 of 155 differ (126260 vs 180975).
- The bug is a signed reinterpretation inside one node and does not depend on timing.
- With a fresh checkpoint the unchanged code verifies (below).

**Verification setup:** worktree `build/wt/fix024b` (branch `fix/024b` = test2/fifo 9f8449048, no code change), FINN_BUILD_DIR `fx/fix024b`, fresh reference estimates. The new checkpoints have `Requant_hls_1` in place of `Requant_rtl_1` (scan: ok).
- `test_transformer_fifo_sizing[ZCU104-transformer_radioml-heuristic_analytical-small]`: **PASSED**. Verified, rtlsim 3051.76 fps = 1.000x the estimate, FIFO 313.4 kbit.
- vision small, same test: stitched-IP rtlsim output **SUCCESS** (`verify_stitched_ip_rtlsim_0_SUCCESS.npy`, 14:28).
  - The test's throughput measurement was still running at hand-off (PID 3661687; first output after ~2.3 M cycles, campaign run ~8 h).
  - The campaign build measured 1.00x with the same FIFO sizer.
- transformer_benchmark small FIFO passes in the campaign (STATUS).

**Fix / action:**
- No code change is needed on any PR branch; the code fix is already landed.
- Campaign, not done here (campaign-owned dirs):
  1. Move the 4 `fx/fifo2/est_transformer_*radioml*/*vision*` dirs above to `fifo2/superseded_024/`.
  2. Rerun radioml/vision FIFO analytical, FIFO rtlsim, folding and DWC. The folding/DWC STATUS cells date from 10-21 (stale checkpoints), even though folding2/dwc2 have since dropped theirs.
- Optional hardening:
  - Land cb8f2ae67 (cache key = source fingerprint). Its cost: every source change rebuilds all caches, which is expensive for resnet50/bert.
  - Or add a cheaper guard against stale caches.

## Scripts
- `scripts/r50_pool_node_check.py` (committed, claude-tools 811ae36c4).
- `../019-transformer-regression/scripts/stale_requant_scan.py <FINN_BUILD_DIR>...`

## Housekeeping
- The first node-check run had a script bug: it set attributes on copies of the node protos, so the code-gen attributes were not cleared. As a result it reused and rewrote files in the finished campaign build's code-gen dirs:
  - `fx/fifo2/code_gen_ipgen_ConvolutionInputGenerator_rtl_21_vwxsmmn3`: wrapper .v regenerated from identical attributes;
  - in that dir, `..._StreamingDataWidthConverter_rtl_153_nv1uk_ow` and `..._Pool_hls_1_3j2wb4nd`: per-node rtlsim `input_0.npy` / `output*.npy` added.
  - There was no HLS rerun and no IP change.
- Nothing deleted.
