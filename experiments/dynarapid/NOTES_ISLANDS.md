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
