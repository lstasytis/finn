# DynaRapid place-and-route for FINN

Experiments with replacing Vivado place-and-route of FINN accelerators by
[DynaRapid](https://github.com/AGS-L/DynaRapid): every dataflow layer becomes a
pre-implemented, relocatable component, and DynaRapid assembles the accelerator
from them according to the ONNX graph.

## Flow

```
dataflow ONNX (all nodes through IP generation, FIFOs/DWCs inserted)
  |
  |  per node, in parallel, content-addressed and cached in a library
  |  (finn.util.dynarapid.components)
  |    1. one-node block design from the node's own IPI commands + adapter
  |       (AXI-Stream -> DynaRapid elastic ports), out-of-context synthesis
  |    2. placement and routing inside a compact pblock (one Vivado run),
  |       port and clock nets left unrouted (DynaRapid GenerateFastPblocks)
  |    3. database of valid relocation sites (RapidWright)
  |
  |  graph: ONNX -> DynaRapid dot file   (finn.util.dynarapid.graph)
  |
  |  DynaRapid GenerateDesign: greedy placement of the components,
  |  stitching, RWRoute of the remaining nets (inter-component, clock,
  |  reset, constants)
  v
routed accelerator checkpoint
```

In the build flow it is used from `step_synthesize_bitfile` (Zynq shell flow): with
`dynarapid_pnr=True` in the build config, `ZynqBuild` implements each compute kernel
with DynaRapid (clock left to the shell), instantiates a wrapper with the stitched-IP
interface whose core is a black box, and an `opt_design` pre-hook inserts the routed
kernel (`read_checkpoint -cell`) and locks it (`lock_design -level routing`), so that
Vivado only implements the shell around it.

Options: `dynarapid_library_dir` (shared, content-addressed component library),
`dynarapid_workers` (parallel component jobs).

## Scripts

| script | purpose |
|---|---|
| `prepare_model.py` | BNN-PYNQ TFC/CNV with reduced PE/SIMD up to IP generation (`--nodes` extracts a subgraph, `--scale` multiplies PE/SIMD, `--part`) |
| `run_experiment.py` | stitched-IP OOC synth + Vivado P&R (`baseline`) vs DynaRapid (`dynarapid`) |
| `run_bitfile_experiment.py` | `ZynqBuild` bitfile with and without DynaRapid |
| `verify_netlist.py` | post-route functional netlist of a DynaRapid design vs stitched-IP RTL (xsi) |
| `bisect_prefix.py` | find the first node where the DynaRapid design stops matching |
| `scaling_study.py` | MVAU -> MVAU with growing PE/SIMD |

## DynaRapid changes

DynaRapid (and its RapidWright submodule) are patched by `fetch-repos.sh` with
`docker/dynarapid/*.patch`. Main changes:

* library generation restored from history and replaced by a single-run generator
  (`GenerateFastPblocks`, compact shapes, speculative parallel attempts); the original
  pin-exposing flow produced functionally wrong FINN components
* arbitrary stream widths and explicit component names (`dcp = ...`) in the dot file
* valid placements from `Module.calculateAllValidPlacements` (routing-aware), exact
  site recovery, library pblocks away from device edges (edge long wires break on
  relocation)
* BRAM resources counted, CARRY8 fix, configurable Vivado, threads, clock period,
  work/library directories, generation region, any part name
* thread-safety fixes around RapidWright device caches (hangs / null clock regions)
* optional largest-first placement and timing-driven RWRoute
* memory-aware parallelism (large devices: one JVM + Vivado per job need ~10 GB)

## Results (KV260 / xck26, 5 ns, 28 parallel jobs, Vivado 2023.1)

Functional check (post-route netlist simulation vs stitched-IP RTL, random inputs):
TFC (10 frames), MVAU->MVAU (4 frames) and the first 8 CNV nodes on the U250 (FIFO,
thresholding, DWC, sliding window, MVAU with BRAM weights; 3 frames) match bit-exactly.
Full CNV was not simulated (4.5M cycles per frame at this folding).

Out-of-context implementation (stitched IP synth + P&R vs DynaRapid):

| design | Vivado | DynaRapid cold library | DynaRapid warm library | WNS Vivado / DynaRapid |
|---|---|---|---|---|
| MVAU -> MVAU | 220 s | 177 s | 4.5 s | +1.29 / +1.43 ns |
| TFC (15 nodes) | 286 s | 250-650 s | 5-7 s | +1.26 / +0.56 ns |
| CNV, weights in LUTRAM (59% LUT) | 1034 s | placement fails | - | +0.22 / - |
| CNV, `ram_style=auto` (90% BRAM) | 630 s | placement fails | - | +0.25 / - |
| CNV `auto` on Alveo U250 (xcu250) | 726 s | 66 min (10 parallel jobs, memory bound) | 280 s, 11 of 37963 nets unrouted | +0.97 / +0.79 ns |

Bitfile (`step_synthesize_bitfile` / `ZynqBuild`):

| design | Vivado | DynaRapid cold | DynaRapid warm | WNS |
|---|---|---|---|---|
| TFC | 713 s | 1347 s | 690 s | +0.64 / +0.54 ns |
| CNV (`auto`) | 1062 s | - | - | +0.11 ns |

Component size (MVAU -> DWC -> MVAU, PE/SIMD x s):

| s | largest layer LUTs | its pblock (rows x cols) / valid sites | Vivado | DynaRapid cold / stitch | WNS Vivado / DynaRapid |
|---|---|---|---|---|---|
| 1 | 1.2k | 20x10 / 45 | 246 s | 259 s / 4.1 s | +0.74 / +0.59 |
| 2 | 2.1k | 20x12 / 45 | 253 s | 249 s / 4.6 s | +0.93 / +0.46 |
| 4 | 6.9k | 45x23 / 40 | 339 s | 412 s / 7.5 s | +0.07 / -0.13 |
| 8 | 22.8k | 85x35 / 32 | 667 s | 798 s / 18.5 s | -0.02 / -2.50 |
| 16 | - | HLS limit (weight word > 8191 bits) | - | - | - |

## Observations

* Stitching pre-implemented components takes seconds; with a warm library the
  accelerator P&R is 40-60x faster than Vivado for the small designs.
* A cold library costs about as much as (or more than) Vivado: every component pays a
  fixed synthesis + place + route overhead (~2-4 min), and the slowest component
  bounds the parallel build.
* In the bitfile flow of small models the Zynq shell dominates Vivado's time; a
  DynaRapid kernel only saves the kernel's share (~3% for TFC).
* Component checkpoints must not keep their out-of-context clock routing: it pins
  relocation to the same position within a clock region (U250: 156 instead of ~57000
  valid anchors for a FIFO), which made CNV unplaceable even on the U250.
* On large devices every component job pays a large fixed cost (device load, placer
  initialisation of minutes) and memory limits parallelism, so a cold library is slow.
* Large layers are the limit of the approach: their pblocks cover a large part of the
  device, have few relocation sites, cannot be packed with the rest of the design
  (CNV on KV260) and their interface logic can end up far from the neighbouring
  components, which non-timing-aware placement turns into long critical paths
  (scale 8: -2.5 ns, timing-driven routing does not help).
