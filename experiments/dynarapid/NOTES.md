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
