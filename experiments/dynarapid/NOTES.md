# DynaRapid integration: status, patches, results

Status snapshot of the DynaRapid place-and-route integration (branch
`feature/dynarapid-pnr`), written 2026-09-25 after the dev container crashed during
the session and was recovered. See `README.md` for the flow description and the full
result tables; this file records what was recovered, which patches are required, and
the state of every experiment.

## Recovery check (2026-09-25)

After the crash (and after adding DynaRapid + a JDK to the Docker image in `fadab3169`)
everything was found intact:

* `deps/DynaRapid` (commit `cf79165`) still holds all modifications; its diff is
  **identical** to `docker/dynarapid/dynarapid-finn.patch` (24 files, +1278/-118).
* `deps/DynaRapid/RapidWright` (commit `edde6618`) diff is identical to
  `docker/dynarapid/rapidwright-finn.patch`.
* DynaRapid was rebuilt with Java 11 after the JDK install (`build/classes/java/main`,
  2026-09-24 22:52); the component library `deps/DynaRapid/library/placedRoutedDCPs`
  (1.2 GB) is present.
* FINN-side code, experiment scripts and all experiment outputs
  (`$FINN_BUILD_DIR/dynarapid_exp`, 1.7 GB) survived. None of it was committed before
  this commit.

Because a fresh `fetch-repos.sh` clones DynaRapid at the pinned commit, **the patches
in `docker/dynarapid/` are the only persistent copy of the DynaRapid bugfixes**.
When changing DynaRapid in `deps/`, regenerate them:

```
git -C deps/DynaRapid add -N . && git -C deps/DynaRapid diff -- . ':!RapidWright' > docker/dynarapid/dynarapid-finn.patch
git -C deps/DynaRapid/RapidWright diff > docker/dynarapid/rapidwright-finn.patch
```

## Installation

| piece | where |
|---|---|
| JDK 11 (`openjdk-11-jdk-headless`, `JAVA_HOME`) | `docker/Dockerfile.finn` (committed in `fadab3169`); a portable JDK also lives in `deps/jdk-11` |
| clone DynaRapid `cf7916512d25` + recursive RapidWright submodule | `fetch-repos.sh` |
| apply `rapidwright-finn.patch`, then `dynarapid-finn.patch` (idempotent: skipped if already applied) | `fetch-repos.sh` (`apply_patch`) |
| set `DYNARAPID_ROOT`, `RAPIDWRIGHT_PATH`, `GRADLE_USER_HOME`; `./gradlew compileJava` if no build exists **or a patch is newer than the build** | `docker/finn_entrypoint.sh` |

The DynaRapid build is optional: a failed build only prints a warning.

## Required patches

### RapidWright (`rapidwright-finn.patch`)

| file | fix |
|---|---|
| `design/DesignTools.java` | `createCeSrRstPinsToVCC`: skip (warn) flip-flop cells placed on a BEL without the requested pin instead of a NullPointerException |
| `rwroute/GlobalSignalRouting.java` | `findCentroid`: skip clock pins whose tile has no clock region instead of crashing |

### DynaRapid (`dynarapid-finn.patch`)

**Library generation (new / restored):**

* `entry/GeneratePblocks.java`, `entry/GenerateDatabase.java`,
  `pblockgenerator/PblockGenerator.java`: library generation entry points restored from
  the DynaRapid history (removed upstream).
* `pblockgenerator/GenerateFastPblocks.java` (new): single-run generator. Implements the
  component once in a compact pblock, then unroutes only port **and clock** nets
  (routed by RWRoute at stitching). Replaces the three-run pin-exposing flow, which
  produced **functionally wrong** FINN components (see verification results). Runs
  speculative attempts in parallel at decreasing utilization and cancels the slower,
  denser attempts after a grace period once one succeeds.
* `pblockgenerator/GenerateShapedPblocks.java` (new): compact, roughly square pblock
  shapes computed from the component's resource needs (instead of tall one-column
  shapes grown row by row), relaxed only on failure, several aspect ratios.

**Correctness:**

* `modules/Pblock.java`: valid placements = sites passing both
  `ModuleInst.getAllValidPlacements()` and routing-aware
  `Module.calculateAllValidPlacements()` (PIPs must exist at the new location).
* `databasegenerator/PblockDatabase.java`: keep only valid sites whose anchor
  (map element side, site index) is recovered exactly from `R#_C#`, otherwise modules
  land on invalid anchors.
* `PblockGenerator`: library pblocks kept away from device edges (edge long wires break
  on relocation).
* `parser/UtilizationParser.java`: BRAM tiles counted (were always 0); CARRY8
  correction sign fixed (`carry8 - carryPresent`).
* `modules/Node.java`, `Input.java`, `Output.java`, `graphgenerator`: dot-file nodes
  may name their component explicitly (`dcp = "..."`) and then keep arbitrary stream
  bit widths (the Dynamatic library only knew 32-bit datapaths).
* `entry/GenerateRouted.java`: drop Vivado's unconnected `[msb:lsb]<bus>` placeholder
  nets from OOC netlists (EDIF 20-100); optional non-flattening and timing-driven RWRoute.

**Thread safety:**

* `graphplacer/GraphPlacer.java`, `graphgenerator/GraphGenerator.java`: pre-fill
  RapidWright's non-thread-safe tile-to-clock-region cache before parallel work
  (it caused hangs and null clock regions), and read module checkpoints one at a time
  (concurrent `Design.readCheckpoint` hangs).

**Configurability (environment variables):**

| variable | purpose | default |
|---|---|---|
| `DYNARAPID_WORK_DIR` | synth/exposed DCPs, Vivado runs, designs | `settings.env` |
| `DYNARAPID_LIBRARY_DIR` | placed-and-routed component library (no release download) | `settings.env` |
| `DYNARAPID_VIVADO` | Vivado binary | `vivado` from `PATH` |
| `DYNARAPID_VIVADO_THREADS` | `general.maxThreads` per run (was a broken `set general.maxThreads 16`) | 16 |
| `DYNARAPID_CLK_PERIOD` | clock period for component P&R and stitched design | 2.5 ns |
| `DYNARAPID_PBLOCK_REGION` | map region used for library generation | whole map |
| `DYNARAPID_PLACE_ORDER=size` | greedy placement of the largest components first | graph order |
| `DYNARAPID_ROUTE_MODE` | `noflatten`, `timing` | flatten, non-timing |

Also: any full part name (was fixed to `xck26` in `GenerateDesign` and the database
header), Vivado runs in the tcl script's directory with a 1 h limit, a registry of
running Vivado processes so speculative runs can be killed, segfaults retried.

## FINN-side changes

* `src/finn/util/dynarapid/`: `components.py` (per-node one-node block design with an
  AXI-Stream to elastic adapter, OOC synthesis, `GenerateFastPblocks`, database; content-
  addressed and cached, parallel, memory-aware worker limit), `graph.py` (ONNX to
  DynaRapid dot, external ports, kernel wrapper Verilog), `flow.py` (end-to-end driver,
  `dynarapid_env`), `tools.py` (Java/Vivado invocation).
* `src/finn/transformation/fpgadataflow/dynarapid_pnr.py`: `DynaRapidPnR` transformation,
  sets metadata `dynarapid_routed_dcp`.
* `make_zynq_proj.py` / `templates.py`: `ZynqBuild(dynarapid=...)` implements non-DMA
  kernels with DynaRapid (clock left to the shell), instantiates a black-box wrapper and
  inserts the routed kernel with an `opt_design` pre-hook
  (`read_checkpoint -cell` + `lock_design -level routing`).
* `build_dataflow_config.py` / `build_dataflow_steps.py`: `dynarapid_pnr`,
  `dynarapid_library_dir`, `dynarapid_workers`, wired into `step_synthesize_bitfile`.

## Results

All runs used Vivado 2023.1 and a 5 ns clock. Outputs are in
`$FINN_BUILD_DIR/dynarapid_exp` (`runs_*.log`, `verify_*.log`, `bit_*.log`). They are not
committed and live under `/tmp`, so copy them if they need to be kept.

### Functional verification (post-route netlist vs stitched-IP RTL, `verify_netlist.py`)

| design | component flow | result |
|---|---|---|
| TFC (15 nodes, KV260) | original pin-exposing library | **mismatch** (1/3, 3/8, 2/6 frames correct across runs) |
| TFC | `GenerateFastPblocks` | match, 10/10 frames |
| MVAU0 alone | pin-exposing library | **mismatch** 0/3 |
| MVAU0 alone | synthesized only / fast flow | match 3/3 |
| MVAU -> MVAU (tfc2) | fast | match 4/4 |
| CNV first 8 nodes (U250, `bisect_cnv/k0_8`) | fast | match 3/3 |
| full CNV (U250) | fast | **not finished**: simulation was at cycle 1.68M of 20M when the container crashed |

### Out-of-context P&R (`run_experiment.py`)

| design | Vivado | DynaRapid cold | DynaRapid warm | WNS Vivado / DR |
|---|---|---|---|---|
| MVAU -> MVAU | 220 s | 177 s | 4.5 s | +1.29 / +1.43 ns |
| TFC | 286 s | 250-650 s | 5-7 s | +1.26 / +0.56 ns |
| CNV KV260, LUTRAM weights | 1034 s | component P&R fails (`component_failed`, 57 of 61 built) | - | +0.22 / - |
| CNV KV260, `ram_style=auto` | 630 s | placement fails | - | +0.25 / - |
| CNV U250, `auto` | 726 s | 66 min (9 jobs, memory bound) | 280 s, 11 of 37963 nets unrouted | +0.97 / +0.79 ns |

### Bitfile (`run_bitfile_experiment.py`, `ZynqBuild`)

| design | Vivado | DR cold | DR warm | WNS |
|---|---|---|---|---|
| TFC | 713 s | 1347 s | 690 s | +0.64 / +0.54 ns |
| CNV `auto` | 1062 s | - | - | +0.11 ns |

With the old library the cold TFC bitfile run failed in synthesis (`bit_tfc_dr_cold.log`).
The `fast` runs work.

### Scaling (MVAU -> DWC -> MVAU, PE/SIMD x s)

s = 1, 2: comparable to Vivado, stitching takes about 4 s. s = 4: -0.13 ns WNS.
s = 8: -2.50 ns (Vivado -0.02). s = 16: over the HLS limit. See `README.md`.

## Open issues / next steps

1. Finish the full-CNV netlist check on the U250 (at 4.5M cycles per frame it needs a
   longer run or fewer frames), or keep bisecting past node 8 with `bisect_prefix.py`.
2. 11 unrouted nets in CNV U250 (warm library): find out why RWRoute leaves them.
3. CNV does not fit the KV260 with DynaRapid: the largest layers have few relocation
   sites and cannot be packed. Options: split large layers, or use shaped pblocks with
   more aspect ratios and `DYNARAPID_PLACE_ORDER=size`.
4. Timing on large components: the placement is not timing-aware, and interface logic
   ends up far from the neighbouring components.
5. Cold-library cost: about 2-4 min fixed per component. On large devices, memory
   (about 10 GB per job) limits parallelism.
6. Add e2e tests (`tests/end2end`, TFC/CNV with `dynarapid_pnr=True`). None exist yet.

---

# Progress log

## 2026-09-25: ZCU104, whole-accelerator DynaRapid flow with a pre-implemented shell

Target: ZCU104 (xczu7ev-ffvc1156-2-e), 5 ns, 32 cores / 125 GB, Vivado 2023.1. Models from
`prepare_model.py --part xczu7ev-ffvc1156-2-e` (same reduced foldings as before).
Outputs in `$FINN_BUILD_DIR/dr_zcu104` (not committed).

### Where the Vivado bitfile flow spends its time (TFC, ZCU104, 700 s)

`run_bitfile_experiment.py` now records the wall-clock time of every ZynqBuild stage:

| stage | s |
|---|---|
| IODMA HLS (two partitions, sequential) | 43 + 44 |
| stitched IP packaging (idma, kernel, odma) | 26 + 45 + 25 |
| MakeZYNQProject: block design, OOC synthesis of its IPs (SmartConnect 108 s, ...), top synthesis 31 s, implementation (place 96 s, route 41 s, bitstream 24 s) | 516 |
| total | 700 |

The kernel-only DynaRapid mode of the previous session cannot win here: it only removes the
kernel's share of place and route, the shell block design, its synthesis and a global
implementation remain.

### New flow (`ZynqBuild(dynarapid={...})`, default `shell=True`; `finn.util.dynarapid.zynq`)

1. ZynqBuild partitions as usual (IODMA insertion, DWCs, partitioning); IP generation of the
   partitions runs concurrently. **No stitched IP is created**, and HLS nodes identical to
   ones of an earlier build (same component hash) reuse their IP (`<dcp>.ip.json` in the
   library).
2. The partitions are merged into one accelerator graph *including the IODMAs*. IODMAs are
   DynaRapid components too: the HLS Verilog is synthesized directly (no block design), and
   each AXI channel (AW/W/B/AR/R of `m_axi_gmem` and `s_axi_control`) is one elastic channel
   (data = packed payload, valid/ready = the channel handshake). The IODMA nodes are called
   `idma<i>` / `odma<j>` in the dot file, so their unconnected AXI channels become top-level
   ports with fixed names.
3. In parallel: the component library (all components concurrently, cached) and the **shell**
   (`finn.util.dynarapid.shell`, cached per board / clock / IODMA interface widths):
   PS + AXI interconnect + SmartConnect + pass-through "bridge" modules named `idma0`,
   `odma0` (AXI interfaces towards the interconnects, the packed channels towards the
   accelerator) + one accelerator cell. The shell is implemented in a pblock below the PS and
   in a 3-column strip next to its fabric interface (EXCLUDE_PLACEMENT, CONTAIN_ROUTING),
   with a placeholder in the accelerator cell; afterwards the placeholder is turned into a
   black box, its boundary nets are unrouted and the static design is locked. The block
   design's `.hwh` is the shell's, with the IODMA names and address map the FINN driver uses.
4. DynaRapid places, stitches and routes the accelerator outside the shell region
   (`DYNARAPID_PLACE_REGION`).
5. Assembly (Vivado): `open_checkpoint` shell, `read_checkpoint -cell` accelerator,
   `route_design` (only boundary + clock nets are unrouted), `write_bitstream`.

Only the assembly touches the full design; the accelerator is never placed or routed by
Vivado as a whole.

#### Things that did not work / had to be fixed on the way

* **DynaRapid map on the xczu7ev**: the map builder dropped every row in which one column
  has no site; the PS is shorter than the die, so 240 of 360 CLB rows were dropped. It now
  drops the (27) columns below the PS instead when that keeps more of the map (360 x 43).
* Hard-coded sites that do not exist on the xczu7ev: HD.CLK_SRC `BUFGCTRL_X0Y2` (component
  runs), `BUFGCE_X0Y8` (OOC designs), the greedy placer's default center `SLICE_X67Y624`.
  Replaced by device-derived choices. Relocation sites outside the map (below the PS) are
  skipped by the database generator.
* **DFX for the shell** (accelerator as a reconfigurable partition): implements fine, but
  pblock snapping removes ~5 full-height columns (next to clocking/configuration columns)
  from the partition, and components with BRAMs can only be relocated vertically (the column
  pattern repeats rarely): the IODMA had 70 anchors, all overlapping a removed column.
  Replaced by the placeholder approach above (no DFX license or rules involved).
* Placeholder with one register driving all outputs: Vivado routes nets from a common
  driver through shared site pins, which is illegal once they become separate nets
  (`Constraints 18-608`). One register per output bit.
* A black box can not go through `opt_design` outside DFX (DRC INBB-3), hence the placeholder.

### Results

**TFC** (17 components incl. IODMAs):

| | Vivado ZynqBuild | DynaRapid flow |
|---|---|---|
| total | 700 s | 672 s with a cold shell (474 s, built once), about 200 s with the shell cached |
| of which | 184 s IODMA HLS + stitched IP, 516 s shell project + synthesis + implementation | 55 s IODMA HLS (parallel), 6 s DynaRapid stitching, 136 s assembly (open shell 30 s, read accelerator 15 s, route 50 s, reports 4 s, bitstream 24 s, final checkpoint 12 s) |
| WNS | +0.943 ns | +0.769 ns |
| routing errors | 0 | 0 |

**Functional check** (`verify_accel.py`, new): xsim testbench with an AXI-Lite master
programming the IODMAs like the FINN driver and AXI memory models; the post-route netlist of
the DynaRapid accelerator plus the shell's bridge modules versus FINN's regular stitched IP
of the same accelerator graph (RTL). TFC, 16 frames of random input: all output bytes
identical (156.0 vs 155.96 us simulated). This covers the IODMA adapters, the AXI channel
packing in the bridges and the DynaRapid routing.

**CNV**: all 61 components build on the ZCU104 and the design places (it did not fit the
KV260). Open: RWRoute leaves 5 nets unrouted (congestion) and WNS is -2.9 ns. The WNS comes
from inside one component: the shape chooser made BRAM-heavy MVAUs very tall and narrow
(21 BRAMs -> 105 x 2 map cells), whose internal BRAM-to-logic wires miss 5 ns by ~3 ns (its
route_design alone ran 16 min trying to close timing). The shape score is now area x
sqrt(aspect deviation) (was area x deviation: 100 x 34 for 33 BRAMs, too wide to place;
then area x deviation^0.25: too tall).

### Cold component builds

Per component (all in parallel): synthesis 80-100 s, pblock place and route 150-300 s
(`place_design` alone ~80 s even for small components: fixed per-phase overhead on the
xczu7ev), database 3-15 s. ~10% of first attempts (target utilization 0.8) fail, and a failed
attempt costs up to ~6 min of congested routing before the next one starts; the slowest
component bounds the build. Now hedged: if an attempt runs longer than 150 s
(`DYNARAPID_HEDGE_S`), the next, less dense attempt starts alongside it and the densest
success is kept. `place_design/route_design -directive Quick` was tried: slower and with
routing errors.

### DynaRapid upstream: where did library generation go?

Library generation (GeneratePblocks, GenerateDatabase, PblockGenerator, the synthesizer
package) was removed from https://github.com/AGS-L/DynaRapid on 2026-03-20 in commits
`9fb2322a5` / `cfb0f2bcd` ("clean up repo"); since then only prebuilt library zips (xck26,
xcvu13p, xczu3eg, Dynamatic components) are released (v0.2.0, v0.3.0). No reason is given
anywhere (commits, README, issues). There is no other branch; the 4 forks and the other AGS-L
repos (incl. DynaRapid-PYNQ-Video-pipeline) do not have it either, and nothing is in
EPFL-LAP/dynamatic. The last commit with the full generator is `b74e30655` (2025-11-18),
which is what the restored generator here is based on. `cf79165` (our pin) is still HEAD.
So there is nothing newer to pull in. Worth asking the author (Andrea Guerrieri,
andrea.guerrieri@ieee.org / @epfl.ch; co-authors at AMD: Chris Lavin, Eddie Hung):
why the generator was removed, whether a newer internal version exists (library format of
v0.2.0+, target-clock / streaming features), and whether a PR restoring it would be
accepted. RapidWright's own pre-implemented module flow (BlockStitcher, PBlockGenerator) is
the maintained alternative for component generation.

### CNV on the ZCU104, continued

* BRAM-heavy components relocate only vertically: right of the PS the xczu7ev map has four
  BRAM columns and almost no repeating column pattern (probe: every 6-column window except
  two pairs is unique). All large MVAUs were generated at the same columns and competed for
  one column band (heights 85 + 125 + 55 + 45 + 30 + 25 > 360 rows). Components with BRAMs /
  DSPs now get two pblock variants at disjoint columns; DynaRapid places CNV after that.
* The relocation database did not know that BRAM cascades (deep weight memories) must stay
  within one clock region: assembly failed with DRC CASC-31. The database now drops anchors
  that split a cascade (dedicated `CASDO*`/`CASO*`, DSP `ACOUT/BCOUT/PCOUT/...` nets) across
  clock regions; for the 33-BRAM MVAU that left 5 of 56 anchors. Components are therefore
  synthesized without BRAM cascades (`-max_bram_cascade_height 1`, component hash version 2).
* CNV Vivado baseline on the ZCU104: **1034 s** (IODMA HLS + stitched IP 267 s,
  MakeZYNQProject 768 s; place 145 s, route 123 s), WNS +0.53 ns.

### Cold builds, resources

* Cold TFC (fresh library, shell cached): **669 s** vs 700 s Vivado - no gain. Library build
  491 s: per component synthesis 72-90 s, pblock P&R 170-394 s.
* Fixed Vivado costs dominate small components on this device: `place_design` of a 15-LUT
  FIFO takes 82 s on an idle machine (32 s placer device model, ~10 s per placer phase),
  unchanged by `-directive Quick/RuntimeOptimized`, `-no_psip`, `-no_timing_driven`, without
  HD.CLK_SRC or CONTAIN_ROUTING. A second P&R in the same Vivado session places in 60 s.
  **Six components placed and routed in one run take 95 s in total** (link 15, opt 8, place
  55, route 14, per-cell checkpoints 3 s) - but `write_checkpoint -cell` drops the routing
  (and HD.PARTITION is not allowed out of context); a batched flow would need RapidWright's
  `DesignTools.copyImplementation` to split the result plus per-component metadata.
* **Memory**: a component P&R run peaks at 4.6 GB, a DynaRapid JVM (device model) at
  2.8 GB. The first cold CNV attempt ran 28 component jobs x (2 variants x hedged attempts)
  = 152 Vivado processes and exhausted the 125 GB + swap (likely the cause of the earlier
  container crash). Now: a machine-wide pool of Vivado slots (lock files under
  `$FINN_BUILD_DIR/dynarapid_vivado_slots`, shared by DynaRapid and FINN,
  `DYNARAPID_VIVADO_SLOTS` to override; default min(cores, 0.51 x free GB / 4.5) = 13 here)
  and component jobs limited to 0.34 x free GB / 3.

### Current results (ZCU104, 5 ns, 32 cores, shell cached)

| | Vivado ZynqBuild | DynaRapid, warm library | DynaRapid, cold library |
|---|---|---|---|
| TFC (17 components) | 700 s, WNS +0.94 | **135 s** (stitch 6 s, assembly 127 s), WNS +0.46 | 665 s |
| CNV (63 components) | 1034 s, WNS +0.53 | **277 s** (stitch 36 s, assembly 238 s), WNS +0.41 | ~1850 s (library 1581 s) |

Warm = all components in the library (same model rebuilt, or a model whose layers were built
before); HLS is skipped then too (IP cache). The shell is built once per board / clock / IODMA
interface (474 s, shared by TFC and CNV). Assembly = open shell 30 s, read accelerator
15-23 s, route 46-137 s, bitstream 25-30 s.

* Assembly routing errors on CNV (2 nets): RapidWright's static-net routing (VCC to a FF
  clock-enable site pin shared with a signal of the relocated component) came in as fixed
  routing. Assembly now clears IS_ROUTE_FIXED on POWER/GROUND nets before `route_design`;
  CNV then routes cleanly and meets timing.
* Functional check after these changes (TFC, 16 frames, verify_accel.py): identical outputs.
  CNV is not simulated (millions of cycles per frame at this folding); its components use
  the same flows (MVAU/SWG/thresholding/FIFO verified bit-exactly on the first 8 CNV nodes in
  the previous session, IODMAs/bridges on TFC).

The cold build is the open problem: with the machine-wide limit (13 Vivado runs, memory
bound) CNV's components need 5251 s of synthesis and 13570 s of pblock P&R in total, i.e.
~1450 s of wall time at best, mostly fixed per-run Vivado cost on small components.

## 2026-09-25 (continued): making cold builds faster

The cold library build is bound by the total Vivado time of all component runs divided by
the Vivado slots (13 here, memory bound). Most of a component's time is fixed per-run cost.

### Batched component implementation (`GenerateBatchPblocks`)

* The pblocks (shapes and variants chosen as before) of many components are packed into the
  generation region without overlap (each keeps its columns, which decide where it can be
  relocated to), the synthesized components are linked into one out-of-context design with
  one pblock (CONTAIN_ROUTING, DONT_TOUCH) per cell, and the design is placed and routed once.
  Six components: 95 s in total instead of ~110 s each.
* Vivado cannot write a cell with its routing outside DFX (`write_checkpoint -cell` keeps only
  the placement; HD.PARTITION is not allowed out of context). RapidWright splits the result:
  `DesignTools.copyImplementation` of the cell into the component's own synthesized design
  (internal nets only: port and clock nets stay unrouted, as in the single flow), work library
  consolidated (the batch netlist has one library per linked checkpoint, `work_c0`, which
  collided between components). A short Vivado pass per batch opens each split checkpoint and
  adds the pblock constraint, utilization report and RapidWright metadata.
* Robustness: 0.6 target utilization in batches (one unplaceable pblock fails the whole
  placement); routing errors attributed per component (conflict / antenna nets by cell prefix;
  pblocks contain their routing, so overlaps are always within one component; per-net
  `report_route_status -of_objects` prints every route tree and was far too slow); failed
  items retried at 0.45 by the batch's own thread; batch time limit 900 s (one congested
  batch routed for >30 min); individual hedged generation as the last fallback.
* Shape fixes found on the way: exact BRAM/DSP need (the utilization headroom for them only
  made BRAM-heavy shapes taller); height/width capped at 8 CLB tiles (165 x 2 map-cell shapes
  were congested and slow to route).
* TFC: 17 components in 188 s (individually ~480 s); functional check (16 frames) identical.

### Faster component synthesis

* RTL nodes (FIFO, DWC, sliding window, thresholding) and single-cell HLS nodes (IODMA, Pool,
  LabelSelect) are synthesized from their HDL directly (no project / IP catalog / block
  design; unused inputs tied to 0 as in the block design). A FIFO run: ~50 s, of which
  `synth_design` is 16 s. MVAU keeps the block design (weight streamer hierarchy).
* Several direct syntheses in one Vivado session (`close_design` + `remove_files` in between;
  results identical to separate runs): the second component took 11.5 s instead of ~45 s.
* Pipelining: batches of components start as soon as enough components are synthesized
  (groups of ~1/6 of the components, at most slots/2 DynaRapid batch JVMs at a time).
  With small groups (6) this was slower (more batches, each with the fixed P&R cost).

### Cold results so far (ZCU104, shell cached)

| | Vivado | DynaRapid cold, per-component | + batched P&R | + direct synthesis |
|---|---|---|---|---|
| TFC | 700 s | 665 s | - | **553 s** |
| CNV | 1034 s | ~1850 s | 1261 s | 1259 s |

CNV cold, batched + direct synthesis: synthesis 362 s, batches 512 s (incl. retries), databases
73 s, stitching 78 s, assembly 188 s, HLS 45 s. Warm builds stay at 135 s (TFC) / 277 s (CNV).

## 2026-09-25 (later): pipelined library build, builder integration - IN PROGRESS

Session paused here (nothing running). State:

* Uncommitted: pipelined `build_library` + `synth_session` (components.py / flow.py),
  builder fixes (assembly writes `utilization.xml`, cells renamed to FINN node names via
  `zynq._name_cells`, `vivado_timing_rpt` metadata used by `step_synthesize_bitfile`),
  new `tests/end2end/test_end2end_dynarapid.py` (+ `dynarapid` marker in setup.cfg). The
  e2e test has not been run yet.
* Cold TFC with the pipelined build (`bit/tfc_pipe`, library `lib11`, shell cached):
  **814 s, worse than the 553 s of `1a6db3645`**. Synthesis is done at 92 s (sessions
  work), but the batched P&R ends at 642 s: groups of 4 components each get their own
  batch run, so there are more, less efficient runs. WNS +0.69, 0 routing errors. Not
  functionally verified yet (`verify_accel.py --accel-dir bit/tfc_pipe/dynarapid`).
* Next: verify tfc_pipe; keep synth sessions but batch P&R as before (all at once, or
  larger groups); CNV cold run; run the e2e test; archive `$FINN_BUILD_DIR/dr_zcu104`
  logs/json to `experiments/dynarapid/results/` (gitignored); commit.
* Note: `--shell-lib` must point at `<lib>/shells` (e.g. `dr_zcu104/lib/shells`).

### Library build pipeline, final state

1. Synthesis: directly synthesizable components in sessions (several per Vivado run),
   MVAUs (block design) individually; all limited by the machine-wide Vivado slots.
2. As components are synthesized: large ones (>= 4 BRAMs or > 5000 LUTs) are implemented
   individually (hedged attempts from 0.8), the small ones in batch groups (~1/6 of the
   components per group, shared place-and-route runs at 0.6, per-component error attribution,
   immediate retries at 0.45, 300 s batch time limit), followed by their placement databases.
3. Individual fallback for anything left.

Other fixes in this round: RWRoute iteration limit 30 in the Zynq flow (2 overlaps persisted
from iteration 16 to 99 on CNV, 80 s; the assembly resolves them); `route_design -directive
Quick` in the assembly routes faster (55 vs 92 s) but misses timing (WNS -0.78), RuntimeOptimized
is the same as the default.

Measurement note: one CNV run (`cnv_sess`) was disturbed by a concurrent TFC build
(`bit/tfc_pipe`, not started from this session) that shared the library and the Vivado slots.

### Results (ZCU104, 5 ns, shell cached)

| | Vivado ZynqBuild | DynaRapid warm library | DynaRapid cold library |
|---|---|---|---|
| TFC | 700 s, WNS +0.94 ns | **135 s**, WNS +0.46..0.77 | **553-618 s** (latest 595 s; one outlier 893 s, see below) |
| CNV | 1034 s, WNS +0.53 ns | **248 s**, WNS +0.69 | **909 s**, WNS +0.18 |

Cold CNV (909 s): HLS of the IODMAs 46 s, library 632 s (synthesis done after 206 s),
stitching 35 s, assembly 194 s. Warm CNV: stitching 55 s, assembly 190 s. Cold TFC (618 s):
library 441 s (synthesis done after 89 s), assembly 124 s.

Cold times vary by +-20 %: which components end up congested (and retried) differs from run
to run; the 893 s TFC run had one batch routing 5 min on a congested small MVAU before the
batch time limit was lowered from 900 to 300 s.

### What limits further gains

* Warm builds: the assembly (open shell 30 s, read accelerator 22 s, route 92 s,
  bitstream 30 s) is now 75 % of the time. Routing boundary + clock nets could be moved to
  RapidWright (populate the shell's black box, RWRoute, Vivado only writes the bitstream), but
  the shell contains encrypted IP (SmartConnect) which RapidWright can only carry through as
  encrypted cells.
* Cold builds: per-run Vivado overhead (~35 s start + device load, ~80 s placer fixed cost on
  the xczu7ev) and the memory bound on parallel Vivado runs (13 here). Batching and synthesis
  sessions amortize most of it; the largest components (BRAM-heavy MVAUs, 3-7 min each) and
  congestion retries now form the critical path.
* These BNN-PYNQ models are small; the global Vivado place-and-route that DynaRapid replaces is
  only ~270 s of the 1034 s CNV flow. The shell pre-implementation removes most of the rest
  (block design, IP synthesis, global implementation), and for larger designs the share of
  global place-and-route grows.

Final functional check (latest flow, cold TFC build, 16 frames): identical to FINN's RTL.

## 2026-09-28: e2e blocker (unplaced CARRY8 slice) root-caused and fixed

Symptom: e2e MLP assembly failed with DRC UNPL-1 (16 cells of
`StreamingDataflowPartition_1_MVAU_hls_0`, cell `n2`). Findings:

* The **library checkpoint** `mvauhlsx159743c61ccc_I55_J11_R10_C7_placedRouted.dcp` was already
  broken: the 16 cells had empty LOC with `STATUS=ASSIGNED` (the earlier check used
  `STATUS==UNPLACED` and missed them). The other variant (J28) was fine.
* In the batch design (`batch.dcp`), Vivado had placed all 16 cells in SLICE_X68Y297 and
  routed with 0 errors.
* The split (`DesignTools.copyImplementation` in `GenerateBatchPblocks.runBatch`) produced a
  checkpoint that, when reopened by Vivado in the metadata step, reported
  `[Constraints 18-4521] Instance c4/.../inputBuf_9_fu_196_reg[14] does not exist` and
  `[Designutils 20-2070] placement information for 1 sites failed to restore`. The metadata
  step then rewrote the checkpoint without that site, so the loss went unnoticed. The site
  contains a LUT route-through cell (D5LUT, rt for DFF2); the written placement still
  references the batch hierarchy `c4/` for it. Re-pointing the cells' EDIFHierCellInst to
  the destination netlist does not help.
* Reproduced standalone (RapidWright harness on batch.dcp): as-is split → 16 cells without
  LOC and 40 nets with routing errors; removing the route-through cells → 0 without LOC and
  1 unrouted net, the same as the known-good J28 variant (Vivado rebuilds the
  route-throughs from the site PIPs; the affected net is ROUTED).

Fix (`GenerateBatchPblocks`): drop route-through cells after `copyImplementation`; the
metadata step now counts cells without LOC and withholds `META_OK` if any are found, so a
bad split falls back to an individual build instead of corrupting the library silently.
`FLOW_VERSION` 3 → 4.

Result (Vivado 2023.1, cold library after the FLOW_VERSION bump): `pytest
tests/end2end/test_end2end_dynarapid.py` **3 passed** (export, build, functional check vs
FINN RTL) in 776 s; DynaRapid flow 530 s (parallel library 387 s, stitch 5 s, assembly
138 s), WNS +0.701 ns at 5 ns, 0 routing errors. TFC `verify_accel.py` regression deferred
to right after the planned switch to Vivado 2024.2 (FINN's documented minimum), where all
checks are repeated anyway.

## 2026-09-28: Phase 2, finn-examples merged

`git merge expanded-finnexamples` (8eb5e4d8d): only conflict `setup.cfg` markers, both kept
(`node_tree_modeling`, `finn_examples`; `dynarapid` untouched). Also brings a small change in
`streamingdataflowpartition.py` (output dtype cast in execute). All models from
`tests/benchmark/models/download_models.sh` are in `tests/benchmark/models/` (gitignored):
tfc/cnv w1a1/w1a2/w2a2, cnv_1w1a_gtsrb, MLP_W3A3 (kws), unsw_nb15-mlp-w2a2 (cybersecurity),
mobilenetv1-w4a4, radioml_w4a4_small_tidy, resnet50_w1a2.
