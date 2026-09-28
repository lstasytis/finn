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
