# RapidWright island flow (branch `feature/rapidwright-islands`)

Crash-safe log of the island-flow work (newest section last). Branch forked from
`feature/dynarapid-pnr` at f99c9346 on 2026-09-30. Machine: EPYC 9554P 128 threads, 755 GB,
Vivado/Vitis 2024.2, `FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build` (/tmp too small).
U55C platform: `PLATFORM_REPO_PATHS=/mnt/labstore/Xilinx/2025.1/Vitis/platforms`
(not under /opt/xilinx/platforms on this machine).

## Goal (user, 2026-09-30)

Bitstreams of FINN models fast via parallel per-node(-group) place and route + stitching,
RapidWright only (no DynaRapid). QoR may drop, functional correctness must stay. 100 MHz.
Order: small CNV/TFC -> VGG10 -> MobileNetV1. No Vivado hacks, no cheating in comparisons.

## 2026-09-30: viability analysis (before any code)

Why not DynaRapid: everything it adds over RapidWright (component library, relocation-site
database, several pblock variants per component, site-based placer) serves *reuse* of
components at other locations. With cold builds as the normal case it is pure overhead. A cold
build needs: parallel OOC P&R at the final location, merge, route the inter-node nets, insert
into the shell. RapidWright has that (`DesignTools.populateBlackBox` = read_checkpoint -cell,
`PartialRouter`). RapidStream (UCLA) is the same idea (floorplan first, islands, parallel
Vivado, stitch) but its open code is an unmaintained U280-only archive (HLS 2019/2020, Gurobi);
the current tool is proprietary.

Measured Vivado per-run floor (isolated, 10 ns, `scratchpad/vbench`, open+opt+place+route):

| part | design | total | place | route |
|---|---|---|---|---|
| xczu7ev | tiny Pool node | 65-69 s | 38 s | 5-9 s |
| xczu7ev | largest CNV MVAU (74 BRAM) | 85-90 s | 48-52 s | 11-18 s |
| xcu55c | tiny IODMA partition (9k nets) | 144-147 s | ~100 s | 12-20 s |
| xcu55c | whole CNV-w1a1 PE=SIMD=1 kernel (13.4k LUT, 215 BRAM36) | 194-210 s | 125-135 s | 32-47 s |

`-directive Quick` / `RuntimeOptimized` change nothing (fixed cost, not per LUT). Consequences:
per-node runs cost ~N x 65 s CPU (why the DynaRapid flow did ~7x Vivado's total work: CNV
2743 s at 4 cores vs 950 s Vivado); for TFC/CNV-size models a whole global P&R is only ~1.5-2
fixed costs, so parallel P&R can save little there; the gain grows with model size. Hence
**islands** (groups of consecutive nodes, K chosen from size and cores) instead of per node.

CNV-w1a1 with all PE=SIMD=1 (`experiments/dynarapid/folding_cnv_w1a1_pe1simd1.json`): HLS MVAU
needs SIMD >= MW/1024 and SIMD | IFM channels -> MVAU_3/4 SIMD=2, MVAU_5 SIMD=4 (+ matching
SWGs). `run_bnn.py` builds bnn-pynq models with FINN's builder (FIFO sizing off: linear chains
cannot deadlock). Frontend 154 s (U55C and ZCU104).

U55C Vitis baseline of that model (FINN default flow, 10 ns): started 00:23, v++ link from
00:37 (IP OOC synth ~13 min, placement at 01:08). Result: see below. NOTE: runs concurrently
with island development builds, so its time is indicative only.

## 2026-09-30: island flow implementation (finn.util.rwislands)

* `device.py`: per-part site map (one Vivado dump, cached `$FINN_BUILD_DIR/rwislands/devices`):
  tile x/y of every SLICE/RAMB/DSP/URAM site. xczu7ev: accelerator region = tile x >= 30 (the
  shell owns x < 30: strip next to the PS + area above the PS), 360 rows.
* `floorplan.py`: node costs from synthesized utilization (LUT + FF/2 + 150*BRAM + 100*DSP),
  linear partition into K contiguous islands (min max cost DP), snake over ~10-column lanes
  (lane 0 bottom-up, lane 1 top-down, ...), rows grown in 5-row steps until the island's
  resources fit at utilization u (LUT u, BRAM/DSP u+0.35), u raised 0.5..0.9 until all fit.
  Pblock ranges per rectangle (an island can continue into the next lane).
* `netlist.py`: island Verilog (components as black boxes, wired) and accelerator top
  (islands as black boxes; IODMA AXI channels as top ports with the shell's names).
* `flow.py`: shell (cached, existing `dynarapid.shell`) || node synthesis (direct HDL in shared
  sessions, MVAUs with weight streamers via one-node block design) -> floorplan -> island P&R
  (one Vivado run each: read_verilog island.v + synth DCPs, link, pblock with CONTAIN_ROUTING,
  opt/place/route) || top synthesis -> `IslandStitcher.java` (RapidWright: populateBlackBox per
  island, PartialRouter on inter-island pins, clock and top-port nets left to Vivado) ->
  assembly (existing `assemble_tcl`: open shell, read_checkpoint -cell, route_design, bitstream).
* Entry: `ZynqBuild(dynarapid={"flow": "islands", "islands": K|"auto", ...})`;
  `run_bitfile_experiment.py --mode islands --islands K --clk 10`.
* `verify_accel.py` accepts island outputs (`rwislands_zynq.json`, stitched accel dcp).

Models at 100 MHz (`$FINN_BUILD_DIR/rwi`): `tfc/`, `cnv/` (prepare_model.py --clk 10, reduced
foldings), `cnv1/` (CNV-w1a1 PE=SIMD=1, ZCU104). Shell 10 ns: `$FINN_BUILD_DIR/rwislands/shells`.

## Results

(appended as they arrive)

### 2026-09-30 01:32: first island bitstream (CNV-w1a1 PE=SIMD=1, ZCU104, 10 ns)

`$FINN_BUILD_DIR/rwi/bit/cnv1_isl_a`, 64 workers, K=auto -> 7 islands (18/7/10/6/3/1/6 nodes).
Fixes needed on the way: HD.CLK_SRC BUFGCE lookup by CLOCK_REGION property (get_sites
-of_objects clock_region returned nothing for most regions); RapidWright does not recognize
Vivado's `black_box "true"` EDIF property as a black box (expects IS_IMPORTED=true or
black_box=1): the stitcher sets IS_IMPORTED before populateBlackBox; the merged netlist must be
consolidated into one EDIF library (else `work` refers to `xil_defaultlib` written after it,
EDIF 20-83 at read_checkpoint -cell).

| stage | s |
|---|---|
| node synthesis (all parallel, 51 DCPs) | 73 |
| 100 MHz shell (one-time, cached afterwards) | 384 |
| island P&R (7 parallel; link 8, opt 14, place 45-50, route 10-15) | 91-104 |
| stitch (RapidWright: read 8 DCPs 1.5 s, populate 1.8 s, RWRoute 229 connections 2.6 s, write 0.9 s) | 9 |
| assembly (open shell 22, read accel 21, route 55, reports 7, bitstream 23) | 128 |

Result: 0 routing errors, WNS +2.774 ns, WHS +0.010 ns. Estimated end-to-end with cached shell
~320 s (stitch/assembly were re-run by hand after the fixes; next runs give the real total).
Assembly route_design: RT build 16 s + init 12 s fixed, 283 unrouted + 130 partial nets, 100
node overlaps -> the serial tail is now mostly Vivado fixed cost.

### 2026-09-30 ~01:50: TFC (reduced folding, 10 ns) island flow + functional check

`rwi/bit/tfc_isl_a`: K=auto -> 3 islands (1/4/12 nodes). Island flow 278 s (end to end incl.
ZynqBuild's IODMA HLS/partitioning 331 s): synth 67, island P&R 94-98, stitch 6, assembly 107
(open shell 21, read accel 19, route 36, reports 5, bitstream 21). Shell cached (same IODMA
widths as CNV-pe1). 0 routing errors, WNS +3.92 ns.
**verify_accel 16 frames: outputs match FINN's stitched-IP RTL** (post-route netlist of the
RapidWright-stitched accelerator + shell bridges).

### 2026-09-30 ~02:00: CNV (reduced folding, 10 ns), assembly variants

`rwi/bit/cnv_isl_a`: 6 islands, island flow 365 s (end to end 417 s; other builds running
concurrently, so indicative): synth 78, island P&R 97-112, stitch 24 (RWRoute left 7 pins,
Vivado finished them), assembly 151. 0 routing errors, WNS +2.595 ns.

Assembly variants on cnv1 (concurrent load): no utilization reports saves ~5 s;
`route_design -directive Quick` saves ~15 s but leaves hold violations (WHS -0.070 ns, THS
-0.737) -> rejected (hold failures are functional failures at any clock).

Frontends at 10 ns done: MobileNetV1 (`rwi/mnv1`, 132 s, 281 nodes: 141 FIFO, 55 DWC, 27
Thresholding, 15 SWG, 15 MVAU_rtl, 13 FMPadding, 13 VVAU_hls, Pool, ElementwiseAdd_hls (no
encrypted IP); estimate 44k LUT, 530 BRAM18, 39 URAM, 684 DSP), VGG10 (`rwi/vgg10`, 483 s).

### 2026-09-30 ~02:15: U55C baseline done; Vivado-native assembly rejected; MobileNet needs AXI-Lite

* U55C (CNV-w1a1 PE=SIMD=1, FINN default Vitis flow, 10 ns): **xclbin written**
  (`$FINN_BUILD_DIR/u55c_test/cnv1/bitfile/bitfile/finn-accel.xclbin`), step_synthesize_bitfile
  5117 s (under heavy concurrent load from island builds: indicative only). U55C bitstream
  generation works on this machine (platform from the 2025.1 install).
* Vivado-native stitching instead of RapidWright (cnv1: open shell, read_checkpoint -cell the
  top netlist, then each of the 7 routed islands, route_design): the 7 island reads took 141 s
  (vs 9 s stitch + 21 s accel read with RapidWright) and route_design aborted ("6 unrouted pins
  that are still reachable", Route 35-8: OOC partition-pin routing of the islands). RapidWright
  stitching stays.
* MobileNetV1: the finn-examples ZCU104 config has runtime_writeable_weights=1 on MVAU_rtl_12/13/14
  - required by FINN: "Layer with URAM weights must have runtime_writeable_weights=1 if
  Ultrascale device is targeted" (URAM cannot be initialized by the bitstream; the driver loads
  the weights over AXI-Lite). A fixed-weights variant is therefore not possible (BRAM would not
  fit: 265 BRAM36 + 39 URAM). -> the island flow must pass compute-node AXI-Lite interfaces
  through the shell (like the IODMA control interfaces). In progress.

### 2026-09-30 ~02:45: pipelined assembly, stitcher speed-up, VGG10 first run (in progress)

* Assembly Vivado now starts when the shell is ready and opens it while the islands are built;
  it waits for a trigger file ("go"/"abort") from the flow. TFC (`tfc_isl_b`): assembly on the
  critical path 107 -> 80 s, island flow 256 s, 0 routing errors, WNS +3.92.
* IslandStitcher: `createMissingSitePinInsts` over the whole design took 36 s on VGG10; now only
  for the top cell's nets (islands' own nets are complete). VGG10 stitch 93 -> 38 s (read 1.5,
  populate 22, names 4.5, site pins 0.9, RWRoute 779 pins 6.3, write 2.8), same result.
* VGG10 (`vgg10_isl_a`, 10 ns): synthesis 129 components; K=auto -> 4 islands (the largest
  MVAU_rtl dominates the max island cost, so the partitioner takes the fewest islands with the
  same maximum). Island P&R 130 / 250 / 289 / 325 s (link 12-24, opt 19-33, place 68-162, route
  28-98). Assembly: read_checkpoint -cell of the stitched accelerator 89 s (Vivado parsing the
  RapidWright-written EDIF of a 98k-LUT design) - now the largest fixed chunk after the islands.
* AXI-Lite pass-through for compute nodes implemented (MobileNet), MobileNet build running.

### 2026-09-30 ~03:10: VGG10 bitstream (10 ns); static-net fix; synthesis scheduling

* Assembly hang on VGG10: `set_property IS_ROUTE_FIXED 0 [get_nets -hier -filter {TYPE ==
  POWER || TYPE == GROUND}]` after reading the accelerator ran > 28 min (hierarchical net query on
  the full 98k-LUT netlist; killed). Needed because the shell's `lock_design -level routing`
  also locks its VCC/GND nets, which the accelerator's static pins join (without it: Route 35-341
  "Fixed routing constraint on net VCCNet conflicts with pin ...", route_design aborts). Fix:
  unfix the static nets right after opening the shell (0.4 s, accelerator still a black box)
  and unlock the accelerator's static routing in the stitcher (`Net.unlockRouting()` on
  GND/VCC). cnv1 re-check: 0 routing errors, WNS +2.774, WHS +0.010 (unchanged).
* **VGG10 island bitstream: 0 routing errors, WNS +3.638 ns, WHS +0.009 ns.** Stages (first run,
  concurrent with MobileNet): node synthesis 203 s (129 components), floorplan u=0.8 (DSP 64 %
  of the device), 4 islands P&R 146 / 267 / 306 / 342 s (island_1 = the single 35k-LUT,
  384-DSP MVAU_rtl), stitch 38 s (re-run with the site-pin fix), assembly (re-run by hand with
  the static-net fix) 460 s: open shell 21, read accel 83, route 284, reports 40, bitstream 32.
  Estimated island flow with pipelining ~1020 s (vs Vivado ZynqBuild 2382 s at 4 ns, Phase 3;
  10 ns baseline pending). The assembly route_design is now VGG10's biggest serial stage: RT
  build 44 s, init 80 s, 668 node overlaps / 561 partial nets initially, rip-up 2 min.
* Synthesis scheduling: MobileNet's Thresholding_rtl nodes (512-1024 channels x 15 steps in
  distributed RAM) take 5.5-9.5 min to synthesize, the 1024-channel ones > 30 min; the
  round-robin sessions serialized several of them. Now only cheap node types (FIFO, DWC,
  FMPadding) share sessions, all others run individually, longest (FINN LUT estimate) first.

### 2026-09-30 ~03:45: VGG10 rerun with all fixes; assembly overlap experiments; MobileNet fixes

* VGG10 (`vgg10_isl_b`, RuntimeOptimized synthesis, pipelined assembly, concurrent with
  MobileNet): **island flow 1014 s, end to end 1068 s, 0 routing errors, WNS +2.801 ns.**
  Synthesis 219 s (critical: the 35k-LUT MVAU_rtl), floorplan u=0.7, 4 islands P&R
  144/223/352/355 s, stitch 40 s, assembly on the critical path 400 s (read accel 82, route
  243 (RT build 44, init 83, rip-up 75), reports 40, bitstream 33).
* Where the assembly router's work comes from (TFC, state after read_checkpoint -cell): partially
  routed nets = the accelerator/shell boundary nets (expected), conflicts = the GND net's
  hierarchical segments. The stitched VGG10 accelerator alone has 0 conflicts, 2 unrouted.
  Experiments: unrouting the islands' clock nets (island Tcl) -> TFC overlaps 41 -> 34, kept;
  unrouting all static routing in the stitcher (FINN_RWI_STATIC=unroute) -> VGG10 still 495
  overlaps / 524 partial nets, route 262 s -> no gain, default stays "unlock".
* MobileNet blockers fixed: (1) a single synthesis session got all 209 cheap nodes (session count
  derived from left-over slots, MobileNet has more heavy nodes than workers); (2) VVAU_hls in
  internal_decoupled mode lists its internal weight stream in1_V as an s_axis interface ->
  dropped in stream_interfaces, BD adapters make only real streams external; (3) AXI-Lite BD
  port names for the W/R/B channels; (4) BD add_files -copy_to refuses leftovers of an
  interrupted run -> component dir cleared before each synthesis.

### 2026-09-30 ~03:45: MobileNet on ZCU104 does not fit rectangular islands; smaller shell; ZCU102

* MobileNet ZCU104 (finn-examples config, URAM MVAUs with AXI-Lite weights): synthesized per node
  152.7k LUT, 123k FF, 254 BRAM36, 69 URAM, 696 DSP (RuntimeOptimized and default synthesis give
  the same LUTs per node type: MVAU_rtl 60k, Thresholding 32k, DWC 31k (55 DWCs), FIFO 12k).
  The accelerator region (x >= 30) has 21.6k slices, 288 BRAM36, 96 URAM (one URAM column).
* Shell region reduced from all columns left of the strip (7200 slices for a 7.3k-LUT shell)
  to INT columns 20-29 on the xczu7ev (3000 slices); the fabric above the PS (x 0-19, rows
  240-359: 4200 slices, 24 BRAM36) is now the first island lane (new shell key via
  `shell_x0`). TFC (`tfc_isl_d`): islands in the above-PS lane, 0 routing errors, WNS +4.26;
  new shell 363 s (one-time).
* Still no fit for MobileNet ZCU104 at any K / lane layout (slabs or 10-column lanes): BRAM 87 %
  and URAM 72 % of the device, both in few columns; each island's rectangle is sized by its
  scarcest memory type, which wastes logic (trace: island needing 31 URAM takes a third of the
  URAM column's height). Decoupling memory from logic (BRAM/URAM runs anywhere) would need
  pblocks without CONTAIN_ROUTING, i.e. inter-island routing collisions at stitch time - not
  pursued now. Same class of problem as DynaRapid's VGG10 on ZCU104 (DSPs).
* MobileNet therefore on the ZCU102 (xczu9eg: 34.3k slices, 912 BRAM36, 2520 DSP, no URAM;
  finn-examples ZCU102 config: no URAM, no runtime-writeable weights). PS on the xczu9eg:
  columns 0-23, rows 0-179 -> PS_BOUNDARY_INT_X 24, SHELL_X0 17. Frontend 133 s.

### 2026-09-30 ~04:10: MobileNet floorplanning attempts, ZCU102 unlicensed, timing batch started

* ZCU102 is not an option here: "A valid license was not found for feature 'Synthesis' and/or
  device 'xczu9eg'". (xczu7ev and xcu55c are licensed.)
* MobileNet ZCU104 floorplan, all offline with the real synthesized node resources (script
  logic in floorplan.py): chain islands with snake lanes (4 or 8 lanes, full-width slabs), a
  best-fit multi-stack allocator (each lane a stack; hardest islands first), class-based
  islands (URAM-using nodes - deep FIFOs and 3 MVAUs - grouped into separate islands) and a 2D
  rectangle packing (any width/position, least scarcity-weighted waste) - none fits for any
  K in 2..48. Needs vs supply: slices 20.1k / 25.8k, BRAM36 257 / 312 (in 3 separated column
  groups of the main region + 24 above the PS), URAM 69 / 96 (one column), DSP 696 / 1488.
  Every rectangle holding a BRAM- or URAM-heavy island spans the logic between memory columns.
  Conclusion: one rectangle per island cannot pack this design on the xczu7ev; it would need
  non-rectangular islands whose memory lies apart from their logic, i.e. no CONTAIN_ROUTING
  and inter-island routing conflicts resolved at stitch/assembly.
* Fallback implemented: when no floorplan fits, the whole accelerator is one island over the
  whole region, without CONTAIN_ROUTING (its parts are separated by the shell). MobileNet then
  gets a bitstream through the same flow, but without parallel place and route. Run queued
  after the timing batch.
* Timing batch 1 started 03:41 (`run_islands_timing.sh`, MODELS="tfc cnv cnv1 vgg10",
  islands then vivado per model, 64 workers, one build at a time, shell cached).
  TFC islands: 309 s end to end, 0 routing errors, WNS +4.264.

#### Timing batch 1 results (ZCU104, 100 MHz, 64 workers, one build at a time, shell cached)

| model | Vivado ZynqBuild | island flow | speedup | K | synth | island P&R | stitch | assembly | WNS islands / Vivado | routing errors |
|---|---|---|---|---|---|---|---|---|---|---|
| TFC | 644 s | 309 s | 2.08x | 3 | 65 | 104 | 4 | 81 | +4.264 / +4.648 | 0 |
| CNV | 891 s | 365 s | 2.44x | 6 | 74 | 114 | 9 | 113 | +3.726 / +1.632 | 0 |

(island times end to end incl. ZynqBuild's partitioning and IODMA HLS, ~55 s; `summarize_islands.py`)
| CNV-w1a1 PE=SIMD=1 | (pending) | 615 s (**invalid**: island_1 split across the shell, see below) | | 7 | 72 | 385 | 6 | 99 | +3.618 / | 0 |

Bug found in the cnv1 timed run: the snake let island_1 continue from the above-PS lane
(x 0-19) into the main region (x 30-40); the two rectangles are separated by the shell, so with
CONTAIN_ROUTING the connections between them are nearly unroutable (route 294 s instead of
~13 s; it did finish with 0 errors). Fixed (04:30): an island restarts in the next lane when
that lane is not adjacent; the snake is tried at all utilizations before the 2D packing.
cnv1 islands to be re-timed.
| CNV-w1a1 PE=SIMD=1 (Vivado) | 811 s | | | | | | | | / +3.657 | |
| VGG10 | (running) | **1107 s** | | 4 | 203 | 370 (152/235/355/370) | 38 | 442 (read 83, route 285, reports 40, bitstream 32) | +3.497 / | 0 |

VGG10 timed island run: a gate-level verify_accel (2 frames, single core) ran concurrently.
MobileNet shell (with the three MVAU AXI-Lite bridges, new region) pre-built untimed: 441 s.
Assembly now writes the bitstream before the reports (VGG10: bitstream ~40 s earlier).
Batch 2 queued: cnv1 islands (re-time), mnv1 islands (single-island fallback) + mnv1 Vivado.
| VGG10 (Vivado) | 2112 s | | **1.91x** vs 1107 s | | | | | | / +3.337 | |
| CNV-w1a1 PE=SIMD=1 (re-timed with the lane fix) | 811 s | 358 s | **2.27x** | 7 | 72 | 130 | 6 | 94 | +2.939 / +3.657 | 0 |

VGG10 functional check (`verify_accel.py`, `vgg10_isl_b`, 2 random frames): post-route netlist of
the RapidWright-stitched accelerator + shell bridges vs FINN's stitched-IP RTL: **outputs match**
(netlist sim ~14 min CPU). Caveat: both outputs are class 0 for both random frames, so this
check is weak (a constant-output fault would pass); real RadioML samples would make it
conclusive. The reference RTL sim first failed ("'swg' is not declared"): verify_accel sorted
the sources, putting swg_pkg.sv after its users - packages now first (also the reason CNV
could not be verified in the DynaRapid branch).

### 2026-09-30 ~06:20: MobileNet timed run hung (fork in threads); fixed

The MobileNet island run hung for 30 min in ZynqBuild's partition preparation (all processes in
futex waits, no output since start): the shell flows prepared the partitions in threads, and
PrepareIP/HLSSynthIP (qonnx NodeLocalTransformation) fork multiprocessing pools from those
threads - the classic fork-in-threads deadlock. Now sequential, like the regular ZynqBuild path
(the Vivado baseline). Earlier timed island runs had concurrent partition preparation (up to
~45 s shorter prep than sequential); their comparison is therefore slightly favourable to the
island flow by at most that. MobileNet timing (islands, then Vivado) relaunched 06:20.

### 2026-09-30 ~06:45: MobileNet single-island fallback (first attempt) - assembly failed

`timing/mnv1_islands` (timed, alone on the machine): floorplan fallback (one island, whole
region = above-PS lane + main region, no CONTAIN_ROUTING): synthesis 154 s, the single island's
P&R 1204 s (link 56, opt 103, place 439, route 547) with 0 routing errors, stitch 39 s, then the
assembly failed: route_design "10161 unplaced non Vcc/Gnd instances" - shell cells (AXI
interconnect/downsizers, reset block) lost their placement when the accelerator was inserted;
the MobileNet shell alone is fine (0 unplaced of 20936). Presumably the uncontained island
routing used shell tiles (e.g. slice route-throughs), which Vivado resolves by unplacing the
shell cells. Fallback changed to one island in the main region only (one rectangle, contained
routing; 88 % LUT / 88 % BRAM / 72 % URAM of that region): run `bit/mnv1_isl_c`, started 06:46
**concurrently with the MobileNet Vivado baseline** (timing/mnv1_vivado, started 06:44) - both
times therefore only indicative.

**VGG10 functional check, stronger (8 frames, input amplitude scaled per frame,
`verify_accel.py --frames 8 --vary-amplitude`): outputs match FINN's RTL for all 8 frames, with
different classes (00 00 00 15 15 12 00 00 hex)** -> the island-flow VGG10 build is verified.

### 2026-09-30 ~07:30: MobileNetV1 island-flow bitstream (ZCU104, 100 MHz)

`bit/mnv1_isl_c` (fallback: one island in the main region x 30-69, CONTAIN_ROUTING; concurrent
with the MobileNet Vivado baseline, so indicative): **bitstream written, 0 routing errors, WNS
+2.70 ns**. End to end 2088 s (ZynqBuild prep ~110 s, island flow 1975 s): synthesis 159 s,
floorplan attempts ~98 s (snake + 2D packing at all utilizations, pure Python - should be
short-circuited when the resource totals already exceed what rectangles can pack), the single
island's P&R 1193 s (link 56, opt 102, place 449, route 545) at ~88 % LUT / 88 % BRAM / 72 % URAM
of the region, stitch 36 s, assembly 470 s (read accel 82, route 312, bitstream 42, reports 32).
No parallel P&R here - the speedup potential for MobileNet on the ZCU104 is limited by the
floorplan (see above). No functional check possible yet: the three URAM MVAUs need their weights
written over AXI-Lite at runtime (verify_accel does not do that) and the design is too large
for gate-level simulation in useful time.
MobileNet Vivado ZynqBuild baseline (timing/mnv1_vivado): started 06:44, still running at 07:30.

MobileNet Vivado ZynqBuild baseline (timing/mnv1_vivado, 10 ns, 64 workers): **5871 s**, WNS
+3.922 ns (its first ~50 min overlapped with the island run). Island flow (single-island
fallback) 2088 s -> ~2.8x, but not like for like: (1) runs overlapped; (2) the island flow
synthesizes nodes with -directive RuntimeOptimized, while the baseline's OOC synthesis of the
compute partition (default directive, whole stitched IP) alone took ~39 min - largely the
Thresholding_rtl LUT-ROM cross-boundary optimization (> 40 min for one 1024-channel node with
the default directive in isolation). A fair split needs the island flow with the default
directive (FINN_RWI_SYNTH_DIRECTIVE="") or the baseline with RuntimeOptimized synthesis.

## Summary (2026-09-30, end of session)

| model (ZCU104, 100 MHz) | Vivado ZynqBuild | island flow | speedup | K | functional check |
|---|---|---|---|---|---|
| TFC | 644 s | 309 s | 2.08x | 3 | verify_accel 16 frames match |
| CNV (reduced folding) | 891 s | 365 s | 2.44x | 6 | gate-level sim too long |
| CNV-w1a1 PE=SIMD=1 | 811 s | 358 s | 2.27x | 7 | gate-level sim too long |
| VGG10 | 2112 s | 1107 s | 1.91x | 4 | verify_accel 8 varied frames match |
| MobileNetV1 | 5871 s* | 2088 s* | ~2.8x* | 1 (fallback) | not yet (runtime URAM weights) |

\* overlapping runs and different synthesis directive, see above. All island builds: 0 routing
errors, timing met. Caveats: island flow uses RuntimeOptimized node synthesis (QoR trade the user
allowed; per-node LUTs were identical to default synthesis on MobileNet's node types, but
synthesis time differs hugely for LUT-ROM thresholds); the shell is cached (one-time 363-441 s
per board/clock/DMA interface set) while the Vivado flow builds its block design each time
(no IP cache in FINN's ZynqBuild); earlier timed island runs had concurrent partition
preparation (< ~45 s advantage), since fixed.

Open / next:
1. Floorplanning of memory-dense models (MobileNet on ZCU104): islands whose BRAM/URAM lies
   apart from their logic would need uncontained routing and conflict resolution at stitch time
   (RWRoute soft-preserve or Vivado); or a larger device (U55C: needs the Alveo per-model link
   flow of the DynaRapid branch's Phase 6).
2. Serial tail: assembly (read accel 20-83 s + route_design 36-312 s, RT build/init fixed cost)
   and the largest island (VGG10: one 35k-LUT MVAU, 370 s). Stitching is cheap (4-40 s).
3. Floorplanner: the 2D packing is slow Python (~100 s on MobileNet before the fallback);
   short-circuit or vectorize it.
4. MobileNet functional check needs runtime weight loading over AXI-Lite in verify_accel.
5. Fair MobileNet comparison (same synthesis directive), core-scaling runs of the island flow.

## 2026-09-30 (morning): U55C with a per-model v++ link (option a)

User decision: island-built compute kernel inserted into the per-model `v++ --link`, start with
TFC and CNV (finn-examples bnn-pynq foldings, U55C, 10 ns; `run_bnn.py --board U55C`).

Implementation (`finn.util.rwislands.alveo`, `alveo_build.py`):
* `flow.islands_and_stitch` (floorplan, island P&R, top synthesis, stitching) is shared by the
  Zynq and the Alveo flow.
* `PrepareForLinking(islands={...})` (builder option `rw_islands_pnr`): IODMA partitions as
  usual (stitched IP + xo); the compute partition: node synthesis -> islands in
  ISLAND_REGION (xcu55c SLR1, tile columns 3-108 = SLICE X4-X170, rows 245-474) -> stitched
  core (module finn_accel_core); kernel .xo = FINN's kernel interface (s_axis_i/m_axis_j,
  stream args as CreateVitisXO) around a black-box core (`kernel_wrapper_verilog`,
  `placeholder_xo_tcl` of the DynaRapid branch).
* `VitisLink`: `[vivado] prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=<hook>`; the hook finds the
  black box, `read_checkpoint -cell`, `lock_design -level placement`, pblock over the island
  region with EXCLUDE_PLACEMENT (no other pblocks, v++'s SLR pblocks untouched).

TFC (`rwu/tfc-w1a1/islands`): compute kernel built in 297 s (synthesis 81, 3 islands P&R
151-176, stitch 40), 0 routing errors, 0 unrouted pins; v++ link running.
Baselines (FINN's Vitis flow, same settings) running concurrently for TFC and CNV.

### 2026-09-30 ~12:45: first U55C island xclbin (TFC), functional-validation phase

* First link attempt: Vivado segfault right after the hook's read_checkpoint (stale Tcl cell
  object used by lock_design) -> the hook re-queries the cell by name. Second attempt:
  **TFC U55C island xclbin written, 0 routing errors, WNS +0.003 ns / WHS +0.009 ns** (platform
  clocks); the hook read the stitched core into level0_i/ulp/StreamingDataflowPartition_1/inst/core
  in 205.6 s (read_checkpoint -cell into the unplaced platform netlist), lock + pblock 2 s.
* All U55C runs so far overlapped (TFC/CNV islands and both baselines) -> times indicative only:
  baselines TFC 4834 s, CNV 5387 s (xclbins written); TFC islands 5641 s.
* v++ link stages, baseline TFC vs island TFC: platform IP synthesis ~12 / ~12.5 min, link+opt
  ~8 / ~15 min (incl. the 3.5 min hook), placement ~22 / ~28 min, routing ~7 / ~8.6 min,
  bitstream ~11 / ~12 min. For a TFC-size kernel the per-model link is almost all platform work;
  the island flow can only shorten the kernel's part. The hook had reserved the whole island
  region (most of SLR1) with EXCLUDE_PLACEMENT - now only the islands' rectangles.
* Resource use (user question): my Vivado runs go through the machine-wide slot limit (81 here:
  min(CPUs, memory)); v++ and FINN's baseline Vivado are outside it. L3 (8 x 32 MB for 128
  threads) and 4 KB pages (THP madvise) make per-process speed drop under heavy concurrency.
  Plan: functional validation in parallel, then every timed run alone.
* **TFC U55C island kernel verified** (`verify_kernel.py`, new: post-route netlist of the
  stitched core in the kernel wrapper vs FINN's stitched-IP RTL of the compute partition,
  stream I/O only): 16 amplitude-varied frames, all outputs match (classes vary), identical
  cycle count (1295).
* **CNV U55C island xclbin written: 0 routing errors, WNS +0.003 ns, WHS +0.002 ns** (6 islands,
  kernel built in 274 s, hook read 210 s). Validation-phase time 5723 s (overlapping runs).
* Sequential U55C timing started 13:25 (`run_u55c_timing.sh`: tfc bitfile, tfc islands, cnv
  bitfile, cnv islands; 64 workers; nothing else running except the single-core gate-level
  simulation of verify_kernel on CNV until it finishes).
* CNV U55C island kernel check (`verify_kernel.py`, 4 frames): outputs match and the cycle count
  is identical (316511 over 4 frames), but all 4 outputs are class 06 -> weak (a stuck output
  would pass). The amplitude scaling treats CNV's unsigned image bytes as signed; a better-varied
  input check follows after the timed runs.

#### U55C timed runs (alone on the machine, 100 MHz, 64 workers; `summarize_u55c.py`)

| run | wall | v++ platform synth | hook | opt | place | route | bitstream | kernel build (islands) | routing errors | WNS/WHS |
|---|---|---|---|---|---|---|---|---|---|---|
| TFC Vitis flow | 4860 s | 740 | - | 182 | 1335 | 425 | 668 | (FINN prep ~990 s total) | 0 | +0.003/+0.009 |
| TFC islands | 5426 s | 742 | 209 | 212 | 1518 | 455 | 698 | 259 (synth 81, 3 islands 171, stitch 6) | 0 | +0.003/+0.009 |

TFC: the island flow is 12 % slower. The kernel is a tiny part of the design, so there is
nothing to parallelize away, and the flow adds the hook's read_checkpoint -cell (209 s into the
unplaced platform netlist) and still ~14 % longer placement of the rest of the dynamic region
(pblock over the islands' rectangles with EXCLUDE_PLACEMENT, locked core). For the U55C the
per-model link is a ~65-75 min platform floor (IP synthesis 12 min, placement 22-25 min, routing
7 min, bitstream 11 min); the island flow can only pay off when the compute kernel's own
synthesis and P&R are a large share (big models).
| CNV Vitis flow | 5458 s | 760 | - | 182 | 1608 | 547 | 698 | (FINN prep ~1100 s) | 0 | +0.003/+0.009 |
| CNV islands | 5562 s | 748 | 215 | 212 | 1427 | 667 | 729 | 276 (synth 87, 6 islands 180, stitch 8) | 0 | +0.003/+0.009 |

CNV: island flow 2 % slower; placement of the rest is faster with the kernel pre-placed
(1427 vs 1608 s) but routing slower (667 vs 547 s) and the hook adds 215 s. Conclusion for
TFC/CNV-size models on the U55C: the per-model v++ link dominates (~75 min of platform IP
synthesis, placement, routing, bitstream) and the island flow cannot shorten it; it only replaces
FINN's stitched-IP synthesis of the kernel (a few minutes for these models). The approach needs
kernels whose own synthesis + P&R is hours (U250-class MobileNet, ResNet50) - or a cached platform
region (option b, nested DFX) to remove the per-model platform work.
* Experiment: is the slow `read_checkpoint -cell` due to RapidWright's DCP (text EDIF)? VGG10's
  stitched accelerator re-written by Vivado (open 67 s + write 44 s) reads into the ZCU104 shell
  in 74 s vs 83 s for the RapidWright DCP (-11 %) -> not worth the conversion; the cost is
  read_checkpoint -cell of a large netlist itself (on the U55C ~210 s for TFC/CNV into the
  unplaced platform netlist).
* **CNV U55C island kernel verified** (`verify_kernel.py --frames 8 --vary-unsigned`, pixel bytes
  scaled per frame): all 8 outputs match FINN's RTL with varying classes (03 06 06 06 06 02 06 06),
  identical cycle count (559787).

Status U55C (end of 2026-09-30): the per-model v++ link with an island-built, placement-locked
compute kernel works end to end for TFC and CNV (xclbins, 0 routing errors, timing met, both
kernels functionally verified at the stream level). It is 2-12 % slower than FINN's Vitis flow
for these small models, because the ~75 min platform work per link stays and the hook adds
~3.5 min (read_checkpoint -cell). Open: hook without EXCLUDE_PLACEMENT pblock (placement effect),
U250-class models (where the kernel's own synthesis/P&R dominates), nested-DFX platform caching.

## 2026-10-01: VGG10 and MobileNet on the U55C; cached platform region (nested DFX)

User: test VGG10, then MobileNet (U55C), cache the platform region.

* Correction: MobileNet ZCU104 runs of 09-30 used FINN's default layer specialization; the
  finn-examples ZCU104 specialize config existed (`tests/benchmark/mobilenet_v1/
  specialize_layers_config/`), I had missed it. Both flows used the same model, so the
  comparison holds. `run_mobilenet.py` now uses the per-board finn-examples configs (U55C:
  U250 configs; standalone thresholds only on ZCU102/104, as in the finn-examples test).
* Nested-DFX feasibility (replaying CNV's v++ impl script up to link_design, then
  write_checkpoint -cell level0_i/ulp, update_design -black_box, pr_subdivide -subcells core):
  **pr_subdivide works** - the core (black box of the island kernel .xo) becomes a
  reconfigurable partition inside the ULP (240 s; ULP netlist save 23 s).
* Cached platform region implemented (`rwislands.alveo`, builder `rw_islands_cache_shell`,
  drivers `--cache-shell`): shell key = platform, clock, IODMA parameters, compute kernel
  stream widths. Miss: the v++ hook saves the ULP, black-boxes it, pr_subdivide, pblock
  (SNAPPING_MODE ON) over the island region for the core partition, reads + locks the core;
  after the link the routed design with the core emptied (update_design -black_box, legal for
  an RP), routing locked and static nets unfixed, plus the xclbin = shell. Hit: no IODMA
  packaging, no v++: open shell, read_checkpoint -cell core, route_design, write_bitstream
  -cell level0_i/ulp, xclbinutil --replace-section BITSTREAM:RAW on the shell's xclbin.
* U55C island region now SLR1 + SLR2 (one rectangle each, lanes per SLR; an island never
  crosses the SLR gap). MobileNet (U250 folding) estimate: 362k LUT, 574 BRAM36, 100 DSP.
* Runs started (validation, concurrent): VGG10 islands + shell build, VGG10 Vitis baseline,
  TFC islands + shell build (then a second TFC run tests the cached path), MobileNet islands +
  shell build, MobileNet Vitis baseline.
* First shell-building link (TFC): the hook worked inside v++ (ULP saved 10 s, pr_subdivide
  260 s, core read 197 s; the core is a reconfigurable partition), but DRC HDPR-29 at
  placement: 101 core cells "outside reconfigurable Pblock" - SNAPPING_MODE shrinks the
  partition pblock to whole clock-region rows (UltraScale+ frames span a clock region), the
  islands in rows 245-299 lay outside. Island region now clock-region aligned (SLR1 rows
  240-479, SLR2 480-719, columns 6-105), core partition pblock 3 columns wider. VGG10/MobileNet
  island runs (old region) stopped and relaunched together with TFC (~12:30).
* HDPR-29 again with clock-region-aligned islands: snapping also drops whole columns (7, 8,
  74) and Laguna-adjacent columns in SLR-boundary clock-region rows. Fix: the partition
  pblock's DERIVED_RANGES are stored per part (`rwislands/data/<part>_core_rp[_v2].json`),
  `island_region` restricts the floorplanner's device to those sites and splits the region at
  site-less columns; `pblock_ranges` emits exact (notched) ranges.
* Region versions (`FINN_RWI_REGION`, default v2, part of the shell key from v2 on):
  v1 = SLR1+SLR2 columns 6-105 (RP rect (3,108,240,719); 68k slices, 960 BRAM36);
  v2 = adds SLR0 clock-region rows 2-3 above the HBM rows, columns 6-107 (RP rect
  (3,110,120,719); 87k slices, 1200 BRAM36, 640 URAM). The running TFC/VGG10 shell builds
  use v1; the TFC cached-hit test must run with FINN_RWI_REGION=v1.
* MobileNet U250 folding (synthesized: 398k LUT, 651 BRAM36, 22 URAM, 100 DSP) did not fit the
  v1 region; on v2 only with 2D packing. User: reduce the folding so it roughly fits ->
  `folding_mobilenet_U250_halfpe.json`: PE of every MVAU_hls halved, VVAU PE halved together
  with its FMPadding/SWG SIMD; first SWG, pool, MVAU_rtl, FIFOs unchanged. The full-folding
  MobileNet baseline (in routing after 1h48) was stopped; baseline + islands rerun on the
  half-PE model (`rwu/mnv1h`).
* VGG10 U55C Vitis baseline (concurrent with other builds, so not a timing result): 7532 s,
  0 routing errors, WNS +0.003, WHS +0.009 (vpl synth 738 s, opt 457, place 1944, route 941,
  bitstream 1094).
* Shell-building links (TFC, VGG10; v1 region) got through the hook (ULP saved 10 s,
  pr_subdivide 265 s, core read ~200-330 s) and placement (no HDPR-29 with the stored derived
  ranges), but TFC failed in route_design: `HPR Routing Violation 18-5229: unlocked site pin
  INT_X74Y299/CTRL_E2 used by power or ground net outside container area`. The cell there
  (SLICE_X117Y299, static column 74, outside the dynamic region's derived ranges) is a ULP
  register of the HBM subsystem's SLR crossing (`hmss_0/.../triple_slr.fwd.slr_middle`), which v++
  constrains by `pblock_dynamic_SLR1` to SLICE X117-X145 / DSP / RAMB columns in tile rows
  240-299 (tiles x 74-92). The core partition covered that area, so the crossing registers were
  squeezed into the left-over notch columns, including static column 74 -> illegal in nested
  DFX. Fix: the core partition's pblock leaves out tiles x 74-92 in rows 240-299 (and, for v2,
  the corridor rows 120-239 the path from SLR0 runs through). Regions are now explicit
  (`ISLAND_REGIONS[part][v] = {"islands": [...], "rp": [...]}`), the shell key always contains
  the region, derived ranges recomputed (`data/<part>_core_rp_<v>.json`) with the replay of the
  link's impl script up to pr_subdivide (hdfx_rp_<v>.tcl). VGG10 shell link stopped (same
  geometry). New v2: islands (6,73,120,239), (6,73,240,479), (74,107,300,479), (6,107,480,719);
  partition (3,73,120,299) + (3,110,300,719).
* MobileNet half-PE islands (old v2 region, 12 islands at LUT 0.6): synthesis 161 s, but 3 of 12
  islands ended with 2-4 nets with overlaps after 7-25 min of routing (localized congestion at
  LUTRAM address pins of SWG window buffers / an MVAU DSP input; global congestion low). For the
  Alveo flow the core's placement is locked, its routing not, and route_design in the link (or
  the cached-shell assembly) re-routes with the whole device -> islands with <= 32 nets with
  overlaps are accepted there (`MAX_ISLAND_OVERLAPS`). Floorplan: at every LUT level, BRAM/DSP
  first with margin, then exact (BRAM-bound islands no longer force a denser LUT level on all);
  for MobileNet still 0.6 (the snake wastes area); 2D packing at 0.5 takes 220 s and scatters
  islands (one 3-column strip) - not used.
* v2 (core partition with SLR0 rows 120-299) failed twice in the shell-building link:
  VGG10 at placement with `Place 30-864 SLLs required = 210, available = 0` for
  pblock_core_rp (the pblock had no LAGUNA sites; fixed: the partition pblock now contains the
  LAGUNA sites inside its rectangles, `device.load_laguna`, shell key version 2), and TFC at
  phys_opt with `Place 30-834 clock partitioning failed ... clock region X6Y2` between the
  BLP's and the ULP's debug-bridge `tck` (locked BUFGCE sources on the same track): with
  SLR0's rows 2-3 given to the core, the ULP's SLR0 logic is squeezed into few clock regions.
  -> v2 is not usable; **default region v1** (SLR1+SLR2 minus the HMSS crossing area).
* Floorplanner: the snake now also tries lane widths 14/20/7 columns and LUT levels in 0.05
  steps (MobileNet half-PE on v2: 0.55 instead of 0.65).
* MobileNet half-PE does not fit v1 with the snake (needs 228k LUT = 42 %, 612 BRAM36 = 66 % of
  v1): its 20 deep FIFOs are URAM (`ram_style: ultra` in the U250 config) and 9 of 12 islands
  need 1-3 URAMs, but only 4 of 18 lanes have URAM columns. With those FIFOs in BRAM it fits only
  at LUT 0.75 (the 0.6-0.7 islands already needed 20-36 min of routing and kept overlaps). The
  BRAM is mostly weights (near the minimum, independent of PE). Per the user ("decrease the
  folding so it roughly fits"): `folding_mobilenet_U55C_quarterpe.json` = PE of MVAU_hls / VVAU
  (with their FMPadding/SWG SIMD) a quarter of U250's, deep FIFOs `ram_style: block`; estimated
  fit on v1 at LUT 0.40-0.45. Baseline and islands restarted on that model (`rwu/mnv1q`); the
  half-PE baseline (1.5 h into the link) was stopped.
* v1 shell-building links after the corridor + LAGUNA changes: TFC failed in phys_opt_design
  with `Place 30-834` (clock region X4Y2: ULP user debug bridge `tck` vs `aclk_kernel_01`
  Clk_Out_Cont, locked BUFGCE sources); VGG10 passed placement but failed in route_design with
  HPR 18-5229 again (`INT_X74Y270`). Isolated with replays of v++'s own impl script up to
  phys_opt_design (sibling dirs of impl_1, variant hook per dir, TFC core; 4 variants in
  parallel, ~45 min): A corridor + all LAGUNA in the rectangles -> Place 30-834 (reproduced);
  B corridor only, C LAGUNA only (old rectangle), D neither -> pass. Cause: A's partition held
  the SLR0/SLR1 LAGUNA columns (tile rows 180-299) although the core never crosses into SLR0.
  Fix: a boundary's LAGUNA column only if all its tiles are inside the partition (v1: only
  `LAGUNA_X0Y240:LAGUNA_X23Y479`, the SLR1/SLR2 crossing). Variant E (that rule) replaying.
* HPR 18-5229 root cause (corrected): not the partition squeezing the HMSS registers as such -
  v++'s `pblock_dynamic_SLR0/1` allow SLICE_X117 (tile column 74) in rows 60-119 / 240-299,
  which is outside the dynamic region's container (its DERIVED_RANGES). In v++'s own flow a ULP
  cell there is fine; with the core as a nested partition, the GND/VCC routing to such a cell
  counts as static routing -> violation. Fix: the shell-building hook prohibits the unused
  sites of those two ranges (`ULP_PROHIBIT`, `set_property PROHIBIT`, placement constraint
  only). Variant F (E + prohibit) replaying through route_design.
* 2D packing bug: candidates could span two touching region rectangles (SLR1 rows 240-479 and
  SLR2 480-719) -> MobileNet quarter-PE got 3 SLR-crossing islands. Candidates now lie inside
  one region rectangle.
* Replays continued: E (corridor + only the SLR1/SLR2 LAGUNA) and F (E + ULP prohibits) still
  fail with Place 30-834 in phys_opt_design (X4Y2, ULP debug-bridge tck vs aclk_kernel_01
  throttling clock); only B (no LAGUNA at all) passed with the corridor - i.e. it is not one
  specific LAGUNA range but some interaction of the partition shape with the ULP's clock
  placement. G = F without phys_opt_design (v++ `STEPS.PHYS_OPT_DESIGN.IS_ENABLED=0`, standard
  option): placement and routing complete (prohibits: 60 + 60 sites), but the DFX DRC at the end
  of route_design reports one `HPR Routing Violation - 9 (18-5239): routing node
  INT_X138Y543/EE1_E_BEG2 used by power or ground net is outside the region of container
  pblock_dynamic_SLR0` (tile column 138, SLR2, outside the island region).
* MobileNet quarter-PE dropped: lower PE makes the VVAUs' threshold memories (tmem 128-512)
  BRAM (30 BRAM36 each, no ram_style for VVAU thresholds in FINN), total 799 BRAM36 (> half-PE's
  612); VVAU weights ram_style distributed did not help. MobileNet now runs with half-PE and the
  per-model link (no nested partition, islands EXCLUDE_PLACEMENT only) on the v2 geometry
  (SLR0 rows 2-3 usable without a partition): kernel ok, 12 islands at LUT 0.55, 351 s.
* H (G + `lock_design -level routing` after the ULP black-boxing, standard DFX recipe) and I
  (H with phys_opt): I fails with Place 30-834 as before; H completes placement/routing but
  the DFX DRC reports the same 18-5239 node (INT_X138Y543). Cells there: SLICE_X220Y54x hold
  `level0_i/ulp/HD_PR_DrivenByBlackBox_InsertedInst_BLP_M_AXI_DATA_C2H_00_*` - LUTs Vivado
  inserted for the ULP's (unused) DMA C2H outputs when the hook black-boxed the ULP; they
  survive pr_subdivide and sit in v++'s 2-column strip SLICE_X220-X221 Y540-599 next to the
  BLP, whose GND routing leaves the container. Removing Vivado-inserted cells would be exactly
  the kind of tool hack the user ruled out.
* **Decision: nested DFX (cached platform region) on the U55C stopped.** Each fix exposes the
  next platform-specific HPR rule (HMSS SLR-crossing pblock column outside the container,
  LAGUNA ownership, ULP clock partitioning in phys_opt, black-box insertions at the BLP
  boundary), each test costs ~1 h of link replay. What works and is used: the per-model v++
  link with the island-built, placement-locked compute kernel (TFC/CNV verified earlier). The
  subdivide/extract/assemble code stays in `rwislands.alveo` (documented as not working on
  xilinx_u55c_gen3x16_xdma_3_202210_1). Replay setup for future attempts: sibling dirs of a
  link's impl_1 with v++'s level0_wrapper.tcl cut after the step of interest and a variant
  opt_pre.tcl/hook (see var_* under vitis_link_proj_vogutg95).
* Now: MobileNet half-PE per-model islands link + baseline running; VGG10 per-model islands
  run started; VGG10 kernel verification (verify_kernel, 4 varied frames) running.
