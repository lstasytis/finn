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
