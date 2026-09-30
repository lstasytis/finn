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
