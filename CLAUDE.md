# Agent onboarding: FINN + DynaRapid (branch `feature/dynarapid-pnr`)

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

Open / next (task file: `experiments/dynarapid/TASK_scaling_and_examples.md`):
1. (done) TFC cold 860 → 591 s: components with >= 3 BRAM tiles are built individually
   (`large_bram` 3 in flow.py; the 3.5-tile TFC MVAU congested its batch).
2. VGG10 (RadioML) Vivado vs DynaRapid cold (`run_vgg10.py`); new layer types MVAU_rtl
   (DSP), FMPadding_rtl, StreamingMaxPool_hls; large layers (PE16xSIMD96).
3. Core-scaling experiment: N = 4/8/16/32 cores via `taskset` + scaled
   `NUM_DEFAULT_WORKERS`/`DYNARAPID_VIVADO_SLOTS`/Vivado threads, Vivado flow vs DynaRapid cold;
   report memory-bound points and serial parts.
4. Then the finn-examples models, MobileNet (DynaRapid only); ResNet50 needs an Alveo shell.

## Conventions

* Commit with the attribution trailer from the system prompt; regenerate the patches with any
  Java change; append results to NOTES.md (dated section), not to this file — update the
  Status section here when the state changes.
* Experiment outputs stay in `$FINN_BUILD_DIR`; small logs/JSON are archived (gitignored) in
  `experiments/dynarapid/results/`.
