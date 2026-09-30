# Agent onboarding: FINN + DynaRapid / RapidWright islands

**Branch `feature/rapidwright-islands` (forked 2026-09-30 from `feature/dynarapid-pnr`):
RapidWright-only island flow, see the section "Island flow" right below; its log is
`experiments/dynarapid/NOTES_ISLANDS.md`.** The rest of this file describes the DynaRapid
flow of the parent branch, whose shell, component synthesis and verification the island flow
reuses.

## Island flow (`finn.util.rwislands`, this branch)

User goal (2026-09-30): FINN bitstreams fast via parallel out-of-context P&R + stitching,
RapidWright only, QoR may drop but functional correctness must hold, 100 MHz; models in the
order TFC/CNV -> VGG10 -> MobileNetV1 (ZCU104). DynaRapid is not used: its library,
relocation database and placer only pay off with component reuse (cold builds are the norm).

```
ZynqBuild(dynarapid={"flow": "islands", "islands": "auto"|K, "out_dir", "shell_lib", "workers"})
  rwislands/flow.py rw_islands_zynq_build:
    shell (dynarapid.shell.build_shell, cached)  ||  node synthesis (synthesize: heavy nodes
      individually longest-first, cheap FIFO/DWC/FMPadding in sessions; synth_design
      -directive RuntimeOptimized, FINN_RWI_SYNTH_DIRECTIVE="" for default)
    -> assembly Vivado starts, opens the shell, unfixes its static nets, waits for a trigger file
    -> floorplan.py: K contiguous islands (min-max cost DP), snake over ~10-column lanes,
       rows grown in 5-row steps to the resources at utilization u (0.5..0.9)
    -> island P&R in parallel (read_verilog island.v + node synth DCPs, pblock
       CONTAIN_ROUTING, opt/place/route, clock net unrouted)  ||  top synthesis (islands = black boxes)
    -> IslandStitcher.java (RapidWright populateBlackBox + PartialRouter on top-cell nets,
       one EDIF library, static nets unlocked) -> trigger "go" -> route_design + bitstream
experiments/dynarapid/run_bitfile_experiment.py --mode islands --clk 10 --islands auto
experiments/dynarapid/run_islands_timing.sh    (timed comparison vs FINN's Vivado ZynqBuild)
```

Models (100 MHz, `$FINN_BUILD_DIR/rwi`): tfc/, cnv/ (prepare_model.py --clk 10), cnv1/ (CNV-w1a1
PE=SIMD=1, run_bnn.py), vgg10/ (run_vgg10.py --clk 10), mnv1/ (run_mobilenet.py). Compute nodes
with AXI-Lite (runtime-writeable URAM weights, MobileNet) are passed through the shell like
IODMA control interfaces. Pitfalls found: RapidWright needs IS_IMPORTED on Vivado black boxes;
merged EDIF must be one library; never query `get_nets -hier` on the full design (use the
shell-only static unfix); Vivado-native read_checkpoint -cell of islands is much slower than
RapidWright stitching; `route_design -directive Quick` leaves hold violations; never fork
multiprocessing pools from threads (PrepareIP per partition deadlocked).

Status 2026-09-30 (ZCU104, 100 MHz, 64 workers, timed one at a time, shell cached; details and
caveats in NOTES_ISLANDS.md):

| model | Vivado ZynqBuild | island flow | speedup | functional check |
|---|---|---|---|---|
| TFC | 644 s | 309 s | 2.08x | verify_accel 16 frames match |
| CNV | 891 s | 365 s | 2.44x | (gate-level sim too long) |
| CNV-w1a1 PE=SIMD=1 | 811 s | 358 s | 2.27x | (gate-level sim too long) |
| VGG10 | 2112 s | 1107 s | 1.91x | verify_accel 8 varied frames match |

| MobileNetV1 | (running) | 2088 s (single island, concurrent with the baseline) | - | not possible yet (runtime URAM weights) |

All island builds 0 routing errors, timing met (WNS +2.7..+4.3 ns). MobileNetV1 (ZCU104): too
dense for one rectangle per island (BRAM 87 %, URAM 72 % in few columns), built with the
single-island fallback (no parallel P&R). ZCU102 not licensed here. Serial floor: synthesis of the largest node,
the largest island's P&R, assembly (read accel + route_design + bitstream, 80-440 s).


Read this first; it is meant to replace re-reading the code. Deeper history and all
measurements: `experiments/dynarapid/NOTES.md` (long; read only the section you need).
Main branch for PRs: `expanded-finnexamples`.

## Goal (the user's framing)

Replace Vivado's global place-and-route of FINN accelerators by DynaRapid: every dataflow
layer is a pre-implemented, relocatable component (out-of-context P&R in a pblock, done
**per node, in parallel**), and DynaRapid only places/stitches/routes the few inter-component
nets (cheap). Assume **cold libraries are the normal case** (folding, FIFO sizing or
quantization change between iterations), so the win must come from parallelism, not caching.
Target: large finn-examples models (MobileNet, ResNet50 take days in Vivado; the user can
supply Vivado baseline runtimes, only the DynaRapid flow needs running for them). Before those:
everything must work on smaller models, and a **core-scaling experiment** must show DynaRapid
runtime dropping with more cores while the Vivado flow does not. Use all cores of the machine.

## Environment

* Container: 32 cores, 125 GB RAM, Vivado 2023.1 (`/mnt/labstore/Xilinx/Vivado/2023.1`),
  JDK 11. Run Claude Code inside `screen`/`tmux` (the user lost a session once); resume with
  `claude --continue`.
* Persistent: the repo (`/home/lstasytis/finn_dev_lstasytis/finn`, host volume) and
  `$FINN_BUILD_DIR=/tmp/finn_dev_lstasytis` (host bind mount, survives container crashes).
  Everything else in the container is lost on a crash.
* **Other containers/sessions can share both paths** and are invisible to `ps`. Once an old
  session kept running builds and editing files concurrently. Before measuring or editing,
  check `git log`/`git status`, file mtimes, and whether `$FINN_BUILD_DIR/dr_zcu104/*.log`
  are still growing. Never share a library dir between concurrent runs when timing.
* `find` is `bfs`: `-newermt` needs ISO timestamps, not "-1 minute".
* Builder is phase-based: `start_step`/`stop_step` take phase names
  (e.g. `phase_convert_to_hardware`), unless `steps=[...]` is given explicitly.

## Architecture (file → function map)

```
ZynqBuild(dynarapid={...})             transformation/fpgadataflow/make_zynq_proj.py
  shell=True (default) -> apply_dynarapid_shell (:417)   whole accelerator incl. IODMAs
  shell=False          -> dynarapid_kernel (:392)       old kernel-only mode (no speedup, legacy)
builder: dynarapid_pnr / dynarapid_library_dir / dynarapid_workers in build_dataflow_config.py;
         wired in build_dataflow_steps.step_synthesize_bitfile (uses metadata vivado_timing_rpt)

util/dynarapid/zynq.py   dynarapid_zynq_build: merge partitions -> vendor-IP check ->
                         [shell || build_library] -> dynarapid_pnr (stitch) -> assembly tcl
                         -> reports (_reports, _name_cells renames n1.. -> FINN node names)
util/dynarapid/flow.py   build_library (:46): cache check, synth (sessions + individual),
                         large comps individually (hedged), small ones in batch groups,
                         databases, individual fallback.  dynarapid_pnr (:298): dot file +
                         Java GenerateDesign (placement, stitching, RWRoute)
util/dynarapid/components.py  component_name (content hash, FLOW_VERSION), direct_sources /
                         direct_synth_tcl (HDL synth w/o block design), synth_tcl (one-node
                         block design, MVAUs), synth_session (several synths per Vivado),
                         build_component (per-component flow), batch_pblocks / batch_databases
                         (Java wrappers), has_pblocks (needs metadata file), vendor_ip_cores
util/dynarapid/shell.py  build_shell (cached per board/clock/IODMA widths), bridge_verilog,
                         assemble_tcl (open shell, read_checkpoint -cell accel, route, bitstream)
util/dynarapid/alveo.py  U55C (Vitis) flow: DR_REGION, kernel .xo packaging, link config/hooks,
                         dynarapid_alveo_build (Phase 6 in the task file reworks it)
util/dynarapid/graph.py  onnx_to_dot (ids n1.., idma<i>/odma<j>), mm_ports, external_ports
util/dynarapid/tools.py  run_vivado / run_java, vivado_slots (machine-wide lock files,
                         VIVADO_GB=4.5, JVM_GB=3.0), dynarapid_env, PART_TO_DYNARAPID
```

Java side: `deps/DynaRapid` (+ `RapidWright` submodule), not in git. Key classes:
`pblockgenerator/GenerateFastPblocks` (one component, hedged attempts), `GenerateBatchPblocks`
(many components per Vivado run, split with `DesignTools.copyImplementation`),
`GenerateShapedPblocks` (shapes), `entry/GenerateDatabase` (relocation sites),
`GenerateDesign` (placement + stitching + RWRoute).

## DynaRapid patches (critical)

A fresh `fetch-repos.sh` clones DynaRapid at the pinned commit and applies
`docker/dynarapid/{rapidwright,dynarapid}-finn.patch`; **the patches are the only persistent
copy of all Java changes.** After editing Java:

```
cd deps/DynaRapid && ./gradlew --no-daemon -q compileJava && touch build/classes/java/main
git -C deps/DynaRapid add -N . && git -C deps/DynaRapid diff -- . ':!RapidWright' > docker/dynarapid/dynarapid-finn.patch
git -C deps/DynaRapid/RapidWright diff > docker/dynarapid/rapidwright-finn.patch
```

Upstream removed library generation in 2026-03 (restored here from `b74e30655`); nothing newer
exists upstream.

## Running things

Scripts in `experiments/dynarapid/` (outputs under `$FINN_BUILD_DIR/dr_zcu104/`):

```
python prepare_model.py --topology tfc|cnv --out $D/tfc --part xczu7ev-ffvc1156-2-e
python run_bitfile_experiment.py --model $D/tfc/dataflow_ipgen.onnx --out $D/bit/<name> \
    --mode vivado|dynarapid --library $D/<libdir>/lib --shell-lib $D/lib/shells
python verify_accel.py --accel-dir $D/bit/<name>/dynarapid --out $D/verify/<name> --frames 16
python run_vgg10.py --model <radioml_w4a4_small_tidy.onnx> --out <dir> --mode frontend|vivado|dynarapid
pytest tests/end2end/test_end2end_dynarapid.py -x -s     # marker: dynarapid
```

* `--shell-lib` must point at the **`shells` directory** (`$D/lib/shells`), else the shell
  (474 s) is rebuilt. Shells are keyed by board/clock/IODMA widths; a new clock = new shell.

### Scaling experiment on another (bigger) machine

```
git clone -b feature/dynarapid-pnr git@github.com:lstasytis/finn.git && cd finn
FINN_XILINX_PATH=<Xilinx dir> FINN_XILINX_VERSION=2024.2 ./run-docker.sh   # fetches deps,
#   applies docker/dynarapid/*.patch, compiles DynaRapid on first start
# inside the container (in screen/tmux):
experiments/dynarapid/run_scaling.sh          # CORES default: nproc, halving down to 4
python experiments/dynarapid/summarize_scaling.py $FINN_BUILD_DIR/dr_scaling/scaling
```

`run_scaling.sh` prepares TFC/CNV and builds the ZCU104 shell once (untimed), then runs Vivado
flow and DynaRapid cold at each core count (taskset, workers, Vivado threads <= 8,
DYNARAPID_VIVADO_SLOTS = min(N, memory bound)); env overrides MODELS, CORES, MAX_SLOTS, D,
PART. On this 32-core / 125 GB container the memory bound is 13 slots. Needs no Alveo
tools. Results of the reference run: NOTES.md "2026-09-29: Phase 4".

* Library layout: `<x>/lib` (component pblocks, `.data`, `.bin.data`, `.ip.json` HLS reuse) and
  sibling `<x>/work` (synth DCPs, batches, designs). Default:
  `$FINN_BUILD_DIR/dynarapid_library/<part>/lib`. Cold measurement = fresh `<x>` dir.
* Functional correctness: `verify_accel.py` (post-route netlist + shell bridges vs FINN
  stitched-IP RTL, xsim). Run it after any change to component/stitching/assembly code.
* Models: finn-examples ONNX from
  `https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-<name>.zip`
  (see `tests/benchmark/models/download_models.sh` on `expanded-finnexamples`); VGG10 copy in
  `$FINN_BUILD_DIR/vgg10/models/`.

## Pitfalls already paid for (don't rediscover)

* Memory, not cores, bounds parallelism: a component P&R ~4.6 GB, a DynaRapid JVM ~2.8 GB.
  Unbounded jobs crashed the container once. Keep everything behind `vivado_slot()`
  (default here: 13 slots). On a bigger server slots scale with free memory automatically.
* Fixed Vivado cost per run dominates small components (~35 s start + device load, ~80 s
  placer on xczu7ev). Batching and synthesis sessions exist to amortize it.
* Encrypted Xilinx IP (HLS float cores `floating_point_v7_1`, e.g. `ElementwiseAdd/Mul_hls`
  from float pre/post-processing) cannot pass through RapidWright: black boxes at assembly
  (DRC INBB-1). Detected up front (`vendor_ip_cores`, status `unsupported_vendor_ip`). Avoid
  float layers in test models (integer input, TopK at the end absorbs output scales).
* A failing component must not take others down: batch split, database runs and pblock
  completeness (`has_pblocks` requires `_placedRouted_0_metadata.txt`) are guarded; keep it so.
* RapidWright picks up every `.edn` in a checkpoint's directory into `_load.tcl`: stale
  encrypted netlists in a shared `work/vhdlSynthDCPs` pollute unrelated components.
* Component checkpoints must not keep OOC clock routing (pins relocation); BRAM/DSP cascades
  must stay inside a clock region (synth with `-max_bram_cascade_height 1`, database drops
  cascade-splitting anchors); BRAM/DSP components relocate mostly vertically → two pblock
  variants at disjoint columns; pblocks away from device edges.
* Placeholder in the shell's accelerator cell needs one register per output bit
  (Constraints 18-608); black boxes cannot go through opt_design outside DFX.
* Assembly: clear `IS_ROUTE_FIXED` on POWER/GROUND nets before `route_design`.
* Changing how components are built → bump `FLOW_VERSION` in components.py (cache key).
* Measurements vary ±20 % cold (which batches congest). Never time two builds at once.
* Vivado 2024.2: grow pblocks with ONE `resize_pblock` call (many small calls silently drop
  sites); batch Tcl must tolerate the spurious link_design error (Designutils 20-50) and a
  failing route_design with overlaps; batch split drops only 5LUT route-throughs.
* U55C / Vitis: the compute kernel is inside the DFX partition `level0_i/ulp` → no black-box
  swap in a routed design (Coretcl 2-1501, no nested HD.RECONFIGURABLE); cells are found by
  ORIG_REF_NAME; never add pblocks around ULP cells or resize v++'s `pblock_dynamic_SLR<n>`
  (HDPR-23, VPL 30-887); DSP/BRAM components relocate only vertically, so the DynaRapid region
  needs height (SLR2+SLR1 for VGG10); use `v++ --remote_ip_cache`.

## Status (2026-09-28, evening)

Toolchain: **Vivado 2024.2** (container restarted, see `restart.md`); 2023.1 results are archived
in NOTES.md, the 2023.1 library in `$FINN_BUILD_DIR/dynarapid_library_v2023.1`. 2024.2 runs live
in `$FINN_BUILD_DIR/dr_zcu104_2024/` (models, `lib/shells`, per-run libraries, logs).

Results, ZCU104, 5 ns, 32 cores, shell cached, 2024.2, FLOW_VERSION 7 (details in NOTES.md):

| | Vivado ZynqBuild | DynaRapid warm | DynaRapid cold | 2023.1 cold |
|---|---|---|---|---|
| TFC | 715 s | 128 s | 591 s | 553-618 s |
| CNV | 981 s | 200 s | 925 s | 909 s |

All with 0 routing errors; TFC verified (verify_accel 16 frames), e2e test 3 passed. CNV cannot
be fully verified with verify_accel (1 frame = ~4 M cycles, ~8 h gate-level xsim; reference
RTL misses the SWG package).

2024.2 changes that were needed (all in NOTES.md, 2026-09-28 sections): Vivado release in the
cache keys; verify_accel drain wait; batch Tcl tolerates the 2024.2 link_design error
(Designutils 20-50) and route_design failure on overlaps; batch split drops only 5LUT
route-throughs; **pblocks built by one resize_pblock call** (2024.2 silently dropped ~40 % of
the sites when a pblock was grown by many small calls - the main cold-build slowdown).

Serial parts of the DynaRapid flow (cap the scaling): IODMA HLS ~45 s, stitching 6-32 s,
assembly 120-170 s (Vivado: open shell, route boundary/clock nets, bitstream).

Core-scaling experiment (Phase 4, ZCU104, 2024.2, done; NOTES.md "2026-09-29: Phase 4"): Vivado
flow flat (CNV 962-1013 s, TFC 703-764 s at 4-32 cores). DynaRapid cold CNV 3135/1731/1083/1178 s,
TFC 1314/920/649/651 s at 4/8/16/32 cores; memory-bound (13 Vivado slots) from 16 cores for
CNV, critical-path-bound for TFC. DynaRapid beats Vivado only for TFC at >= 16 cores (-9 %).
Extrapolated CNV ~505 s at 64 cores given ~6-7 GB RAM per core. To repeat on a bigger machine
see "Scaling experiment on another machine" above.

Core-scaling experiment, second machine (Phase 4b, details in
`experiments/dynarapid/NOTES_SCALING.md`): EPYC 9554P, 64 cores / 128 threads, 755 GB. The
library's batching was tuned for 13 slots (~12 cores busy on the big machine); now sized per
machine (`batch_plan` in flow.py: batch size from a makespan model with CPU load, pools from
memory/slots, adaptive batch time limit; >= 2-BRAM components and large ones individually,
hedged attempts from slots). Results (s, cold):

| N | CNV Vivado | CNV DynaRapid | TFC Vivado | TFC DynaRapid |
|---|---|---|---|---|
| 128 (SMT) | 894 | 635 | 652 | 444 |
| 64 | 890 | 612 | 651 | 614 (variance; 483 at 32) |
| 16 | 878 | 1376 | 641 | 774 |
| 4 | 950 | 2743 | 697 | 1379 |

Vivado flat; DynaRapid 1.45x faster than Vivado at 64+ cores, crossover ~32 cores, 128
threads no better than 64 cores. Floor now = slowest component P&R + serial assembly (~150 s).
Weak spot: 16 cores (a small CNV MVAU fails its batch and is rebuilt at the end). All 0
routing errors; TFC verified. On this machine `FINN_BUILD_DIR=/home/lstasytis/finn/build/
finn_build` (/tmp too small), repo at /home/lstasytis/finn, no screen/tmux (setsid nohup).

U55C (VGG10): Vitis baseline 9566 s (WNS +0.003 ns). DynaRapid library + stitching work on the
xcu55c (region SLR2+SLR1); the cached-shell assembly is blocked by DFX rules. **Next: Phase 6 in
the task file (per-model v++ link with the locked DynaRapid kernel) - a fresh agent implements
it; all artifacts to start from are listed there.**

Open / next (task file: `experiments/dynarapid/TASK_scaling_and_examples.md`):
1. Scaling experiment done on both machines. Optional: batch robustness (NOTES_SCALING.md
   "Open"): speculative lower-utilization retry of long batches, immediate individual
   fallback, 2 parallel attempts at <= 16 slots.
2. **Phase 6: U55C per-model v++ link** (user decision 2026-09-30; supersedes the cached-shell
   plan of 2026-09-29), then VGG10 timed runs and `run_u55c_vgg10.sh` for the user's 128-core
   server; then the finn-examples models on U55C.
3. Builder: an `accel_dynarapid_failed` result still ends the build with rc=0 (should raise).

## Conventions

* Commit with the attribution trailer from the system prompt; regenerate the patches with any
  Java change; append results to NOTES.md (dated section; core-scaling results to
  NOTES_SCALING.md), not to this file — update the
  Status section here when the state changes.
* Experiment outputs stay in `$FINN_BUILD_DIR`; small logs/JSON are archived (gitignored) in
  `experiments/dynarapid/results/`.
