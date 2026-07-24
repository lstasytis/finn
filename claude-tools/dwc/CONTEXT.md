# DWC — Generalized StreamingDataWidthConverter

## Goal
Make one DWC component handle, in a single node, all three of:
- **width conversion** (fold/unfold between producer and consumer stream widths),
- **zero-padding** (output has more elements than input — pad tail with zeros),
- **cropping** (output has fewer elements — drop the tail),
including **non-multiple and coprime** width ratios. The legacy RTL DWC only does
integer-ratio width conversion; the generalized variant is HLS and adds padding/cropping.

This is a **dependency for the folding optimizer** — folding wants to pad channel
counts to rounder values, which needs a DWC that can pad.

## Branch / PR
- Feature branch: **`feature/generalized-datawidthconverter`** (fork: `origin`).
- The upstream **HLS-side** work (the `StreamingDataWidthConverterGeneralized_Batch`
  kernel in `finn-hlslib/streamtools.h`) is merged into what `dev` points to.
- The FINN-side PR is a **single squashed commit** on top of the feature branch's
  merge-base with `dev`, containing only the 5 source files below (no scratch, and
  `fetch-repos.sh` excluded — bump `HLSLIB_COMMIT` to the upstream commit that has
  the merged kernel instead of the fork).

## Key source files (the PR)
- `src/finn/custom_op/fpgadataflow/streamingdatawidthconverter.py` — abstract op:
  `in_shape`/`out_shape` (not one shape), `execute_node` does per-frame pad/crop.
- `src/finn/custom_op/fpgadataflow/hls/streamingdatawidthconverter_hls.py` — HLS
  backend; `get_ap_int_max_w` bumped for the `inWidth+outWidth` intermediate buffer.
- `src/finn/custom_op/fpgadataflow/hls/iodma_hls.py` — IODMA uses the 4-template-param
  generalized kernel.
- `src/finn/transformation/fpgadataflow/insert_dwc.py` — `InsertDWC` picks **rtl** for
  integer-ratio, no-pad/crop conversions, **hls** (generalized) otherwise.
- `tests/fpgadataflow/test_fpgadataflow_dwc.py` — see Tests.

## Design notes
- The generalized HLS kernel uses a single `inWidth+outWidth`-bit register with a
  constant-shift emit plus a **log-depth barrel** (units of `G = gcd(in,out)`,
  `ceil(log2 OutLanes)` stages) for the non-multiple placement — avoids a full-range
  barrel shifter. For **integer ratios it collapses to the same shift-register datapath
  as the plain DWC**, so it costs ~the same on the multiple case (LUT-wise) and only
  differs on non-multiple/padding.
- **Padding/cropping is applied per frame** (per innermost row), not globally across
  the whole flattened stream. The test comparison must therefore compare per frame.
- Full design log: [`DWC_WORK.md`](DWC_WORK.md).

## Tests
- `test_fpgadataflow_dwc` — parametrized over width/pad/crop/coprime configs incl. a
  **padding-stress** block, in cppsim + rtlsim. Value check compares the non-padded
  region **per frame** (`y.reshape(-1,out_last)[:, :k] == x.reshape(-1,in_last)[:, :k]`).
- BIPOLAR configs skip the value check (cppsim packing quirk); INT2/INT4/BINARY are
  fully checked.
- Locally: `pytest tests/fpgadataflow/test_fpgadataflow_dwc.py -k cppsim` (rtlsim needs
  a working verilator/pyverilator, usually unavailable here).

## Resource analysis (this folder)
Real Vivado OOC synth (xc7z020) of RTL vs old-HLS vs generalized, LUT **and** FF, for
increasing input width / output width / both. Key finding: the RTL up-sizer is a shift
register, so growing the output width adds **flip-flops but ~no LUTs** (invisible on a
LUT-only axis). Generalized matches RTL/HLS on the multiple case and beats the old HLS
LCM-cascade on the non-multiple case (which fails to synth once `LCM > 8191`).
- `make_tex.py` → `dwc_resource_analysis.tex` (3×2 groupplot fig, LUT|FF columns).
- `collect_vivado.py` / `collect_ff.py` → `plotdata.json` / `plotdata_ff.json`.
- `add_point.py` / `add_point2.py` — synthesize extra data points and merge into the JSON.
- `ref_plain/` (old HLS), `ref_gen/` (generalized), `ref_lcm/` (old non-multiple LCM
  cascade), `ref_rtl/` (RTL core) — per-variant synth harnesses.
- `lut_harness.py`, `tb_dwc.cpp` — earlier C-sim / LUT harnesses (superseded by the
  pytest padding tests, kept for reference).

## NEXT: finn-examples benchmark testing (2026-07-22) — not yet started
Goal: show the generalized DWC lowers **LUT** vs the default rtl/hls DWCs, per model, at
**real synthesis**. Replicate the folding pattern (see [`../README.md`](../README.md) START
HERE and [`../folding/folding_eval.py`](../folding/folding_eval.py)):
1. `git checkout -b dev-generalized-datawidthconverter dev-expanded-finnexamples`;
   `git cherry-pick <origin/feature/generalized-datawidthconverter commit>` (single commit
   `e59b1f39b`, base is very old — expect conflicts in `insert_dwc.py`,
   `streamingdatawidthconverter*.py`, `iodma_hls.py`; the HLS kernel is upstream in dev now
   so `fetch-repos.sh` should NOT repoint HLSLIB). Get the per-op DWC test passing first
   (`pytest tests/fpgadataflow/test_fpgadataflow_dwc.py -k cppsim`).
2. Write `dwc_eval.py` here: build each finn-examples model to stitched IP with the DWC
   variant chosen by `InsertDWC` (rtl for integer ratios, generalized-hls otherwise), run
   **regular OOC synth** (`stitched_ip_gen_dcp=True`, no P&R) and parse per-instance LUTs of
   the `StreamingDataWidthConverter*` rows from `finn_design_partition_util.rpt` (the
   hierarchical report lists each node). Compare generalized-HLS vs the default the model
   would otherwise get. (Standalone per-DWC OOC synth harnesses already exist under this
   folder — `ref_gen/`, `ref_plain/`, `ref_rtl/`, `ref_lcm/` — for isolated width sweeps.)

## Gotchas
- `fetch-repos.sh` in the feature branch repoints `HLSLIB` to a fork — **do not** ship
  that in the PR; the kernel is upstream now, pin the upstream commit instead.
- RTL DWC requires integer width ratio (asserts otherwise); non-multiple/pad/crop must
  route to the generalized HLS variant via `InsertDWC`.
