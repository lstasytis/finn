# Task: DynaRapid core-scaling experiment and finn-examples models

Context: `CLAUDE.md` (repo root, read fully first). History/measurements:
`experiments/dynarapid/NOTES.md` (read only the section you need).

## Why

The claim to demonstrate: with a **cold** component library (the normal case: folding, FIFO
sizing or quantization change between iterations), DynaRapid's per-layer, out-of-context
place-and-route parallelizes across cores, while FINN's regular Vivado flow (global P&R of
the whole design) barely speeds up with more cores. The user may move this container to a
much larger server based on this experiment, and eventually run MobileNet/ResNet50 (days in
Vivado; the user can provide Vivado runtimes for those, only DynaRapid runs are needed).

## Ground rules

* Correctness before timing: every DynaRapid bitfile you report must have 0 routing errors,
  and `verify_accel.py` must match FINN's RTL (at least once per model/config; CNV-sized
  models can use fewer frames).
* Never run two timed builds at the same time; check for foreign activity first (see
  CLAUDE.md Environment). A cold run = fresh library dir (`<x>/lib` + `<x>/work`), shell
  cached (build shells once per board/clock beforehand and exclude them from timings, but
  report their one-time cost).
* Stay within memory: leave the `vivado_slot()` machine-wide limit in place.
* Record every result as it arrives in a dated NOTES.md section (a crash must not lose it);
  keep CLAUDE.md "Status" current; commit after each phase (attribution trailer per system
  prompt; regenerate the DynaRapid patches if Java changed).
* If a phase is blocked for more than ~2 h of debugging, write down what you found and move to
  the next phase that does not depend on it, then report.

## Phase 0: preflight (short)

`git status`/`git log -5`, confirm nothing else is running, `screen` session in use,
DynaRapid built (`$DYNARAPID_ROOT/build/classes/java/main`). Uncommitted work from the previous
session should already be committed; if not, commit it first.

## Phase 1: blocker — e2e test (`tests/end2end/test_end2end_dynarapid.py`)

Symptom (deterministic, also with a clean library): assembly fails with DRC UNPL-1, 16
unplaced CARRY cells (plus unplaced terminals / partial routes) inside one batched MVAU
component (`StreamingDataflowPartition_1_MVAU_hls_0`, cell `top_i/accel/inst/n2`). Its library
checkpoints (`<lib>/mvauhlsx159743c61ccc/*_placedRouted.dcp`) have **no** unplaced cells
(verified in Vivado). So placement is lost in DynaRapid relocation/stitching
(`GenerateDesign`, `work/designs/accel/accel_routed.dcp`) or in the assembly's
`read_checkpoint -cell`.

Suggested steps: (1) open `accel_routed.dcp` in Vivado, list unplaced primitives → which
side loses them; (2) compare the relocated anchor vs the library placement of those CARRY8
cells (CARRY8 site/BEL mapping at the new anchor, the CARRY8 correction in
`UtilizationParser`, RapidWright relocation of cells in the lower/upper half of a CARRY8);
(3) build the same component individually (non-batched) and see if the problem persists →
batch split (`DesignTools.copyImplementation`) vs relocation. Iterate on the stitching step
alone (warm library, `dynarapid_pnr` only) — minutes, not a full build. When the test passes
(`pytest tests/end2end/test_end2end_dynarapid.py -x -s`), re-verify TFC with
`verify_accel.py` (regression check) and commit.

## Phase 2: merge finn-examples

`git merge expanded-finnexamples` (main branch; adds `tests/benchmark/*` configs and custom
steps; only `setup.cfg` overlaps: keep both markers). Download models with the URLs in
`tests/benchmark/models/download_models.sh` into `tests/benchmark/models/` (gitignored); VGG10
is already in `$FINN_BUILD_DIR/vgg10/models/`.

## Phase 3: VGG10 (RadioML) on ZCU104, all cores

`experiments/dynarapid/run_vgg10.py` (frontend once, then `--mode vivado`, then
`--mode dynarapid` with a fresh `--library`). 4 ns clock → new shell (build it first, time it
separately). New layer types for this flow: MVAU_rtl (DSP), FMPadding_rtl,
StreamingMaxPool_hls; large layers (PE16×SIMD96). Expect problems with large components
(few relocation sites, long builds, timing) — investigate, fix or document. Report: total
time, per-stage breakdown (HLS, library: synth / batches / large components / databases,
stitch, assembly), WNS, routing errors, verify_accel result.

## Phase 4: core-scaling experiment (main deliverable)

Models: CNV (known-good, `$FINN_BUILD_DIR/dr_zcu104/cnv/dataflow_ipgen.onnx`) and VGG10.
Core counts N ∈ {4, 8, 16, 32} (skip 4 for VGG10 if it exceeds ~6 h; say so).

For each N, run the whole build under `taskset -c 0-$((N-1))` and scale all knobs to N:
`NUM_DEFAULT_WORKERS=N` (FINN / Vivado `-jobs`), DynaRapid `--workers N`,
`DYNARAPID_VIVADO_SLOTS=min(N, memory bound)`, Vivado `general.maxThreads` ≤ N (for the
Vivado flow e.g. via `~/.Xilinx/Vivado/Vivado_init.tcl` or `-tclargs`; check what
`templates.py`/`make_zynq_proj.py` set, and restore afterwards). Runs:

* Vivado flow (`run_bitfile_experiment.py --mode vivado`, or `run_vgg10.py --mode vivado`)
* DynaRapid cold (fresh library per run, shell cached)

Report per N: total wall time, stage breakdown, peak memory (e.g. `/usr/bin/time -v` or
sampling `free`), whether the Vivado-slot limit (memory) rather than N was binding. Also
compute from the logs the total CPU/wall work of all component jobs (sum of synth + pblock
runs) and the serial parts (IODMA HLS, stitching, assembly), and give an Amdahl-style
extrapolation for larger core counts / more memory. Present as a table plus a plot (PNG in
`experiments/dynarapid/results/`, and the table in NOTES.md). Be explicit about which points
are memory-bound on this 125 GB machine.

## Phase 5: more finn-examples models (DynaRapid)

Order: cybersecurity-mlp, gtsrb, kws, bnn-pynq variants (configs exist for AUP-ZU3/Pynq-Z1;
target ZCU104 with those foldings, adjust only if a model does not fit), then
**mobilenet_v1 (ZCU104 config exists)**. First check each model for layers with encrypted
Xilinx IP (`vendor_ip_cores`; float pre/post-processing) — the flow rejects them; if a model
needs them, report it rather than working around it silently. For small models also run the
Vivado flow at 32 cores; for MobileNet run only DynaRapid (the user supplies Vivado times).
**ResNet50** only has an Alveo U250 config (Vitis flow); the DynaRapid shell flow is
Zynq-only — do not start it; report what an Alveo shell would require.

## Deliverables

1. Passing e2e test, committed.
2. VGG10 and scaling results (table, plot, extrapolation) in NOTES.md + a short summary in
   CLAUDE.md Status.
3. Per-model results table for the finn-examples models (time, WNS, routing errors,
   verified yes/no, blockers).
4. A final message to the user: results, what limits scaling, recommended server size, open
   issues.

## Update 2026-09-29 (user decision)

VGG10 does not fit DynaRapid's per-component pblocks on the ZCU104 (64 % DSPs, pblocks 90 % of
the region, see NOTES.md). User decision: the core-scaling experiment (Phase 4) runs on the
ZCU104 with CNV and TFC; **everything after it (VGG10, the finn-examples models incl.
MobileNet) targets the Alveo U55C** (`xcu55c-fsvh2892-2L-e`, platform
`xilinx_u55c_gen3x16_xdma_3_202210_1`, Vitis 2024.2; Alveo work is authorized by this).

### Plan: DynaRapid on the U55C (Vitis flow) - SUPERSEDED (2026-09-30, see Phase 6)

The cached-shell variant below was implemented up to the assembly and is blocked there
(NOTES.md "2026-09-29 (evening)"); kept for reference.


FINN's Vitis flow (`alveo_build.py`: CreateVitisXO, VitisLink) links three kernels per model -
`idma0` (HLS IODMA, m_axi + s_axilite), the compute partition (RTL kernel, AXI-Stream only) and
`odma0` - with `stream_connect`, HBM via `sp=`, and `v++ --link` implements the platform's
dynamic region (ULP, DFX) on top of the locked static region (`hw.xsa`: `hw_bb_locked.dcp`).
The ULP also holds encrypted IP (HBM memory subsystem, SmartConnect), so it must stay in Vivado.
Mapping of the Zynq design:

1. **Shell (cached per platform / clock / DMA widths):** `v++ --link` with the real IODMA
   kernels and a placeholder compute kernel (registered AXI-Stream loop, one register per output
   bit, as on Zynq); keep the routed checkpoint (`--save-temps`,
   `_x/link/vivado/vpl/prj/prj.runs/impl_1/*_routed.dcp`) and the xclbin (metadata sections).
2. **Per model:** DynaRapid builds the compute kernel's components (library on the xcu55c part)
   and stitches/routes it within a place region inside one SLR next to the kernel's DMAs
   (SLR crossings out of scope at first).
3. **Assembly:** open the shell checkpoint, `read_checkpoint -cell <compute kernel>`, route the
   boundary, `write_bitstream -cell` for the ULP (partial bitstream), then
   `xclbinutil --replace-section BITSTREAM` on the shell's xclbin.
4. Functional check: XRT hardware is not available here; verification via the post-route
   netlist simulation of the compute kernel against FINN's RTL (as verify_accel, streams only).

Steps: (U1) FINN Vitis baseline on U55C for TFC (v++ link time, toolchain check);
(U2) xcu55c part in DynaRapid (PART_TO_DYNARAPID, library map region per SLR, memory per
Vivado run on the larger part); (U3) Alveo shell + assembly; (U4) TFC/CNV end to end, then
VGG10 and the finn-examples models. Heavy U55C runs only in gaps between timed scaling runs.


## Phase 6 (2026-09-30, user decision): U55C with a per-model v++ link - NEXT AGENT STARTS HERE

Context in one paragraph: DynaRapid on the Alveo U55C cannot use a cached, routed shell as on
Zynq, because the compute kernel sits inside the platform's DFX partition (`level0_i/ulp`) and
Vivado refuses to black-box / replace a non-partition cell in a routed design (details and the
three things tried: NOTES.md "2026-09-29 (evening)"). User decision: **per-model `v++ --link`**
in which the DynaRapid-placed-and-routed compute kernel is read into a black-box core before
`opt_design` and locked, and Vivado places and routes only the rest of the ULP (IODMAs, HBM
memory subsystem, interconnect). The old ZCU104 "kernel mode" did exactly this insertion
(`make_zynq_proj.py:129-146`: `read_checkpoint -cell $c <routed dcp>` + `lock_design -level
routing $c`, core found by `ORIG_REF_NAME || REF_NAME`) - reuse that pattern.

What exists and is reusable (all on `feature/dynarapid-pnr`):
* `src/finn/util/dynarapid/alveo.py`: `DR_REGION` (xcu55c: SLR2+SLR1, corners `SLICE_X4Y707` /
  `SLICE_X170Y252`, DynaRapid place region `"12,467,3,108"` = top,bottom,left,right map
  rows/cols), `placeholder_xo_tcl` (RTL kernel packaging that v++ accepts; keep
  `auto_family_support_level level_2`), `link_config` (FINN's connectivity), `pre_place_tcl`
  (tile-rectangle site collection in ONE `resize_pblock` call), `dynarapid_alveo_build`
  (library + stitching parts are good; shell/assembly parts are to be replaced),
  `shell_key`, `build_shell`, `assemble_tcl`, `package_xclbin` (obsolete after this phase).
* `experiments/dynarapid/run_alveo_dynarapid.py`: takes the model saved after FINN's Vitis
  bitfile step, `link_kernels()` returns the compute kernel model + the kernel list (IODMA `.xo`s
  from `PrepareForLinking`).
* `src/finn/util/dynarapid/graph.py`: `kernel_wrapper_verilog(..., black_box=True)` = wrapper
  (`ap_clk`, `ap_rst_n`, `s_axis_*`, `m_axis_*`) around a black-box `finn_accel_core`
  (`shell.CORE_MODULE`), reset active low (`.rst(ap_rst_n)`, as the Zynq shell).
* `tools.py`: `PART_TO_DYNARAPID` / `PBLOCK_REGION` have the xcu55c (library region
  `"260,10,459,100"`, inside SLR1).
* DynaRapid Java: `GreedyPlacer` recenters the design into the place region when the part's
  default center site is outside it (needed on the xcu55c; patch regenerated).

Artifacts on disk (`$FINN_BUILD_DIR=/tmp/finn_dev_lstasytis`, host mount), usable to start
without rebuilding anything:
* VGG10 U55C frontend: `vgg10_u55c/run/frontend` (`run_vgg10.py --board U55C --mode frontend`).
* Baseline (FINN Vitis flow, 9566 s, WNS +0.003 ns): `vgg10_u55c/run/vivado`; its model after
  linking, **input for the DynaRapid driver**:
  `vgg10_u55c/run/vivado/intermediate_models/step_synthesize_bitfile.onnx`; baseline v++ link
  dir: `vitis_link_proj_2zr82mqr` (config.txt, logs).
* Cold VGG10 component library on xcu55c (122 components, 0 fallbacks):
  `vgg10_u55c/dr0/lib/lib` + `vgg10_u55c/dr0/lib/work`.
* **Already stitched and routed compute kernel** for the SLR2+SLR1 region:
  `vgg10_u55c/dr0/lib/work/designs/accel/accel_routed.dcp` (stitch 156-195 s; RWRoute stops at
  its 30-iteration cap with a few overlaps, as on Zynq - Vivado must finish those; with a locked
  core this needs attention, see risks).
* Platform IP cache for `v++ --remote_ip_cache` (138 IP OOC runs, 74 MB):
  `vgg10_u55c/dr0/lib/shells/ip_cache` (link step 2: 24 → 10 min with it).
* `vgg10_u55c/dr0/lib/shells/vshell*`: obsolete cached shells of the superseded variant.

Steps (verify each before the next; record results in a dated NOTES.md section; commit after
each working step):
1. **Black-box compute kernel .xo** (per model): `kernel_wrapper_verilog(<kernel name>,
   CORE_MODULE, ins, outs, black_box=True)` (ins/outs = `graph.external_ports(compute_model)`),
   packaged with `placeholder_xo_tcl` (same kernel name and stream args as FINN's
   `CreateVitisXO`). Check: package_xo succeeds; the kernel's OOC synthesis inside v++ accepts the
   black box.
2. **Link hook** (`[vivado] prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=<tcl>`): find the core
   (`get_cells -hier -filter {NAME =~ */<compute inst>/* && ORIG_REF_NAME == finn_accel_core}`;
   REF_NAME is uniquified by the platform's per-IP synthesis), `read_checkpoint -cell $core
   <accel_routed.dcp>`, `lock_design -level routing $core`, then `pblock_dynarapid` over the
   DR_REGION sites (reuse the site collection of `pre_place_tcl`) with EXCLUDE_PLACEMENT, core
   added to it. Do NOT create other pblocks or resize v++'s `pblock_dynamic_SLR<n>` (HDPR-23,
   VPL 30-887). Check on the first attempt with the existing `accel_routed.dcp` (one link,
   ~45 min with the IP cache): link log shows the core filled and locked, opt/place/route run,
   0 routing errors, WNS at 4 ns, xclbin written. **This is the go/no-go test; if the locked-core
   insertion cannot be made to work in ~2 h, stop and report to the user.**
3. **Orchestration**: rework `dynarapid_alveo_build`: vendor-IP check → library (`build_library`,
   parallel) → stitching (`dynarapid_pnr`, `place_region=DR_REGION[part]["place"]`, size-order
   retry) → black-box kernel .xo → per-model `v++ --link` (FINN's `config.txt` via `link_config`,
   the hook, `--remote_ip_cache <shared dir>`, `[vivado] synth.jobs/impl.jobs` = cores, `-o
   finn-accel.xclbin`) → reports (timing summary, route status of
   `_x/link/vivado/vpl/prj/prj.runs/impl_1`) into `dynarapid_alveo.json` with per-stage times.
   Remove the placeholder-shell / assembly / xclbinutil code paths (build_shell, assemble_tcl,
   package_xclbin, shell_key) or keep them clearly marked as unused.
4. **Correctness**: xclbin with 0 routing errors, timing met at 4 ns (or report the WNS);
   functional check of the DynaRapid compute kernel against FINN's RTL - post-route netlist
   simulation of the compute kernel with stream I/O only (adapt `verify_accel.py`, which drives
   IODMAs over AXI; here drive `s_axis_0`/read `m_axis_0` directly) with as few frames as VGG10
   allows; if one frame is too long for gate-level xsim (CNV needed ~8 h per frame), report it.
5. **Timed runs here** (nothing else running): DynaRapid cold (fresh library dir, IP cache warm =
   the realistic iteration case) and warm (library cached); stage breakdown (library synth /
   batches, stitch, kernel .xo, v++ link steps) vs. the 9566 s baseline.
6. **Deliverable for the user's 128-core server**: `experiments/dynarapid/run_u55c_vgg10.sh`
   (frontend if missing → Vitis baseline → DynaRapid cold → DynaRapid warm; slots from free
   memory like `run_scaling.sh`; v++ jobs = cores; prints a summary table). Test it end to end
   here once, commit, then tell the user it is safe to run there. (The ZCU104 `run_scaling.sh`
   is already there and tested in its original form.)

Risks to check early:
* Vivado may refuse or disturb a locked, pre-routed cell inside the DFX partition during
  opt/place/route (the Zynq kernel mode had no enclosing partition).
* RWRoute leaves a few overlaps (30-iteration cap); with the core's routing locked Vivado cannot
  fix them → either raise the RWRoute cap for this flow, or lock only placement
  (`lock_design -level placement`) and let Vivado finish the core's routing.
* Other ULP nets routing through the DynaRapid region; the SLR1/SLR2 crossings inside the
  compute kernel at 250 MHz. Fallback if timing fails: keep the kernel within SLR1 by generating
  more pblock variants on other column bands (the SLR1-only region failed only because all MVAU
  variants sit on two column bands).
* The per-model link is a serial chunk (~40-60 min estimated); only the compute kernel part
  scales with cores - report this honestly.
