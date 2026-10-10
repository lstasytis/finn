# Related work: parallel / split FPGA compilation (survey 2026-10-10)

Survey by a research subagent (Fable); extracted paper texts were in the session scratchpad.
[V] = read in the primary source, [S] = search snippet only, [I] = inference.

## Island flows (closest to ours)

- **RapidStream 1.0** (Guo et al., FPGA'22 best paper; github.com/UCLA-VAST/RapidStream) [V]:
  TAPA dataflow PEs assigned to 32 equal islands on a U250 (8x4, ~2x2 clock regions, >50K LUTs),
  one anchor register per inter-island net in thin anchor regions (3 empty CLB columns), anchors
  placed by an LP (20x faster than Vivado's placer), islands P&R'd by Vivado in parallel,
  RapidWright stitch. **Clock: a clock trunk pre-routed to every island entry point with locked
  buffer delay taps** (without: hold violations on all 32 islands). Final step: 5-10 % of nets
  conflict -> Vivado route_design (~25 min of ~2 h) or a single-thread RWRoute partial route
  (~6 min, no hold model -> anchors on the negative edge). 5-7x vs Vivado.
- **RapidStream 2.0** (TRETS 2023; Guo thesis ch. 5.8-5.11) [V]: **no global route**: inter-island
  nets routed first on a skeleton (shell + anchors + island boundary cells), pruned to stubs at
  virtual partition pins; islands are DFX reconfigurable partitions routed in abstract shells
  against those pins (nested DFX: anchors + PCIe/HBM static). U280, 6 islands: 5.1-7.5x vs
  Vivado, ~2x vs RS1.0 (which needed ~3 h of conflict fixing). Rules: one empty row between island
  and anchor region; shell-used SLL/Laguna sites prohibited.
- **RapidStream IR** (ICCAD'24): slot grid (1/4 die), parallel synthesis with black boxes 2.49x.
- **AutoBridge** (FPGA'21): slot floorplanning + pipelining, still one Vivado P&R.
- **RapidPnR** (FCCM'25 poster): netlist-level partition into a grid of islands, inter-island
  sources moved to boundary regions, parallel P&R + merge: 1.6-2.5x (the ceiling of plain
  partition-and-merge).

## Page / overlay flows (fixed grid, pre-built interconnect, no global step)

- **PRflow** (Park, Xiao, Magnezi, DeHon, FPL'18): ZU9EG, 31 PR pages of ~3.9K LUTs + packet NoC
  (20 % of the device); full rebuild 30 -> 7 min.
- **DW** (Xiao, Ahmed, DeHon, FPT'20): pipelined point-to-point links through PR switchboxes
  instead of a NoC, pre-defined partition pins; 10x throughput, Fmax ~187 MHz; page boundaries
  block DSP/BRAM cascades.
- **PLD** (ASPLOS'22): U50, nested DFX inside the Vitis shell, 22 pages, abstract shells per page
  ("essential"); ~18K-LUT pages for ~95 % efficiency; 4.2-7.3x vs Vitis.
- **HiPR** (FPL'22/TRETS'23): per-function PR regions on a 1-column x 1-clock-region tile grid,
  abstract shells, parallel: incremental 3.4-5.6x; first compile 15-67 % slower.
- **Hierarchical PR pages** (FPT'22): nested DFX merges single pages into double/quad pages;
  2.2-5.3x. Critique of RapidStream: 2x2-CR islands too big; global inter-island routing ~30 min.

## Pre-implemented blocks

- **DynaRapid** (FPL'24): pre-P&R'd component library, greedy placement, RWRoute: 20x vs Vivado
  Quick, 1.8x LUTs, routing dominates; only with reuse (library build ~10 h).
- HMFlow / TFlow (BYU / Virginia Tech): hard macros, 10-50x.
- RAW'25 (Damian-Kosterhon, Koch, Kosterhon, Petrica): block sizes for quantized CNNs in a
  RapidWright block flow.

## Engines and the final step

- Vivado route_design: <= 8 threads; RapidStream measured ~2 cores average over a 14 h Vivado run.
- RWRoute (FPT'21): 4.9x vs Vivado, no hold model. Parallel RWRoute (Zang et al., GLSVLSI'24): 2x.
- Abstract shells (UG909, WP533): RMs of several RPs in parallel; 2-10x on Alveo platforms.
- byteman (Manchester): US+ bitstream merge/relocation. No open bitstream generation for US+
  (Project U-Ray incomplete); everybody calls Vivado write_bitstream (full or partial).

## What the literature supports (for our flow)

1. Registered island boundaries (all flows) - we have them (register slices at both ends).
2. **The clock must be fixed before island routing** (RS1.0 trunk + locked taps; RS2.0 via DFX).
3. Granularity ~4-18K LUT pages (PRflow, PLD), 1 column x 1 clock region tiles (HiPR);
   RS1.0's >50K-LUT islands are criticised as too coarse.
4. Fixed grids win on compile time; nested DFX merges adjacent pages for large functions
   (= a layer taking consecutive snake tiles).
5. Direct nearest-neighbour links beat NoCs for dataflow.
6. **Any post-hoc global route caps the speedup**; order-of-magnitude gains only with
   pre-routed interfaces (RS2.0, DW/PLD) or reuse (DynaRapid, HMFlow).

Ranked recommendations: (1) pre-route inter-island links before island P&R (RS2.0 pattern),
(2) islands as DFX partitions with abstract shells and partial bitstreams in parallel (removes the
whole-design load/route/bitstream), (3) split dominant layers to <= ~1 clock region, (4) empty
margin between tiles and link channels, prohibit shell-used SLL/Laguna sites, (5) RapidWright-only
final route only with a fixed clock skeleton and hold margin, (6) parallel top synthesis.
