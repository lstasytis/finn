# DynaRapid core-scaling notes

Scaling experiments of the DynaRapid flow (cold library) against FINN's Vivado flow, split
out of NOTES.md so they do not collide with the VGG10 / U55C notes there. The first-machine
experiment (32 cores, 125 GB) is in NOTES.md "2026-09-29: Phase 4".
Driver: `run_scaling.sh`; tables: `summarize_scaling.py`; plot: `plot_scaling.py`.

## 2026-09-29: Phase 4b, scaling on a 128-thread server; library plan sized for the machine

Machine: AMD EPYC 9554P (1 socket, 64 cores / 128 threads; CPUs 0-63 are distinct physical
cores, 64-127 their SMT siblings, so N=128 adds SMT only), 755 GB, Vivado 2024.2, same
container image. `$FINN_BUILD_DIR` moved to the 1.7 TB repo volume (`build/finn_build`,
/tmp had 43 GB). `run_scaling.sh` defaults: CORES 128 64 32 16 8 4, MAX_SLOTS 83 (memory).
Shell + model prep untimed (CNV prep 4 min, shell run 14.5 min).

### Series 1: flow as tuned for the 32-core machine (`dr_scaling/scaling`)

| CNV, N | Vivado | DynaRapid cold | DR library | DR synth done | DR stitch | DR assembly | DR avg cores | DR peak mem GB |
|---|---|---|---|---|---|---|---|---|
| 128 | 894 | 820 | 588 | 109 | 24 | 151 | 12.4 | 177 |
| 64 | 890 | 874 | 632 | 110 | 24 | 163 | 11.4 | 176 |
| 32 | 888 | 774 | 536 | 164 | 34 | 151 | 11.2 | 102 |

All 0 routing errors, WNS +0.2..+0.6 ns. Vivado flat. DynaRapid does not scale past 32 cores:
**~12 cores busy on average**, although 83 Vivado slots were free. Cause: fixed constants in
`build_library` (flow.py): `group = max(6, min(16, (n+5)//6))` (~6 groups, CNV 11 components
each, split in 2 Vivado runs of 5-6), and the batch pool `min(6, slots // 2)` shared by the
groups and the 7 large (individual, hedged) MVAUs - 12 tasks on 6 workers, so the 5th group
only started at 284 s and the large MVAUs finished at 200-518 s. Batch timeline (n128): a
1-component batch 140 s, 5-6-component batches 183-278 s, i.e. the fixed cost dominates.
Also: `avail_memory_gb` used free pages only (log said "limiting parallel component jobs to
58" with 740 GB MemAvailable), and `os.cpu_count()` ignores taskset.
(N=32 caveat: the driver was stopped during this run; ~8 Vivado runs started 14:07:52-14:08:00
without the maxThreads cap.)

### Change (commit 274c7e2d): `batch_plan` in flow.py

Components per batched Vivado run k (1..8) minimize the makespan estimate
`ceil(ceil(n/k)/slots) * (120 s + 20 s * k)` (tie: larger k); group pool = min(JVMs by
memory, slots); per JVM 1-3 runs; Vivado threads per run = clamp(2*CPUs / concurrent runs,
1, 4); batch time limit (`DYNARAPID_BATCH_TIMEOUT_S`, was a fixed 300 s) =
max(200, 1.6 * expected run time * max(1, 2/threads)). CPUs from process affinity, memory
from MemAvailable. Plans: 128/64 cores k=1 (limit 224 s); 32: k=2; 16: k=4; 8: k=7; 4: k=7,
2 runs per JVM; the old 32-core/13-slot machine: k=5 (as tuned before).

Test (TFC cold, 128 cores, `dr_scaling/test_plan`): 677 s (library 559, stitch 8, assembly
109), 0 routing errors, WNS +0.53 ns, **verify_accel 16 frames match**. 16 of 17 components
done by 259 s; the critical path was one IODMA (`iodmahlsx18978fb7e0ad`) whose 1-component
batch congested at util 0.6, hit the (then 300 s) limit and was retried at 0.45 (done at
559 s). The adaptive limit was added after this test (224 s at k=1). Next improvement: a
speculative lower-utilization attempt for batches that run long while slots are free
(Java, GenerateBatchPblocks), instead of waiting for the limit.

### Further changes found during series 2

* a1d8cc6b: CPU load in the plan. First plan at 16 cores: CNV 1301 s, the 4-component batches
  ran 1.7x their idle time (runs + large components + syntheses > CPUs), one hit the 320 s
  limit and its component was rebuilt individually at the end. Now load factor
  `max(1, 1.3 * concurrent runs / CPUs)` (fitted: median batch time / idle time = 1.0 at 56
  runs on 128 CPUs, 1.17 at 27 on 32, 1.7 at 14 on 16) in the makespan estimate and the
  limit (2x expected, >= 300 s); Vivado threads per run not oversubscribed.
* 00df3e7c: components with >= 2 BRAM tiles are built individually (hedged). The TFC input
  IODMA (2418 LUTs, 2 BRAM) congests at batch utilization 0.6 even alone (every run): TFC
  at 128 cores 734 s, IODMA done at 559 s, everything else by 256 s. Affects one IODMA each
  in TFC and CNV.
* 2bf9be14: parallel pblock attempts of the large components = min(3, slots // 10) (was
  CPUs // all components = 1 for TFC at 32 cores: attempts 0.8 -> 0.6 -> 0.45 ran one after
  another with the 150 s hedge delay; TFC 760 s at 32 vs 477 s at 64).

### Series 2: machine-sized plan (`dr_scaling/scaling_v2`, final code 2bf9be14)

All points with the final code except CNV 8/4, which ran with a1d8cc6b: identical plan
(parallel attempts 1 in both versions at <= 16 slots), only the CNV IODMA (2 BRAM tiles) went
through the batch path there, without problems (0 fallbacks). CNV Vivado 128/64/32 reused
from series 1 (the change does not touch the Vivado flow). Results of the first plans in
`scaling_v2/plan1` (and `results/scaling_128t/scaling_v2_plan1.txt`).

| N | CNV Vivado | CNV DR cold | DR library | DR synth done | DR avg cores | DR peak GB | TFC Vivado | TFC DR cold | DR library | DR avg cores |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 (SMT) | 894 | **635** | 377 | 94 | 36.9 | 385 | 652 | **444** | 270 | 11.9 |
| 64 | 890 | **612** | 365 | 92 | 23.5 | 273 | 651 | 614 | 439 | 8.1 |
| 32 | 888 | 825 | 566 | 269 | 15.7 | 254 | 648 | **483** | 311 | 9.0 |
| 16 | 878 | 1376 | 1128 | 374 | 6.5 | 141 | 641 | 774 | 603 | 5.2 |
| 8 | 868 | 1650 | 1419 | 855 | 4.6 | 110 | 635 | 899 | 730 | 3.9 |
| 4 | 950 | 2743 | 2502 | 1871 | 2.9 | 55 | 697 | 1379 | 1199 | 2.4 |

All DynaRapid runs 0 routing errors, WNS +0.5..+0.8 ns; fallback 0 except CNV 16 (1).
Stitch 31-52 s (CNV) / 6-9 s (TFC), assembly 146-158 s (CNV) / 109-120 s (TFC) at every N.
TFC final build at 128 cores: verify_accel 16 frames match. Plot:
`results/scaling_128t/scaling_v2.png` (`plot_scaling.py <scaling dir> <png>`).

Reading:
* Vivado flow flat (CNV 868-950 s, TFC 635-697 s, ~1.2 cores average).
* DynaRapid scales from 4 to 64 cores: CNV 2743 -> 612 s (4.5x), TFC 1379 -> 444-614 s.
  Speedup over Vivado at 64-128 cores: CNV 1.45x (612 vs 890), TFC 1.47x at 128 (444 vs 652;
  64 cores 614 s is run-to-run variance of the hedged large MVAU `mvauhlsx2798c8f16fa2`:
  done at 291 s in one run, 438 s in the other). Crossover with Vivado at ~32 cores.
* 128 vs 64: no gain for CNV (SMT only; peak memory 385 vs 273 GB).
* What limits it now (CNV, 64 cores): synthesis ~92 s (longest block-design MVAU synth) +
  the slowest component P&R ~270 s (large MVAUs, hedged; a congested batch) + stitch ~40 s +
  assembly ~154 s (serial: open shell, route boundary and clock nets, bitstream) + IODMA HLS.
  Floor of this flow on xczu7ev ~550-600 s for CNV, ~400 s for TFC; per-component work no
  longer is the bottleneck at >= 64 cores.
* The 16-core points are the weak spot: CNV 1376 s (slower than 8 cores' per-core rate)
  because the small MVAU `mvauhlsxaca6f60ab9bc` (763 LUTs, 0 BRAM, 35 CARRY8) fails its
  batch with routing errors at 0.6 and 0.45 in both 16-core runs (with different neighbours),
  holds its batch 668 s (limit + retry) and is then rebuilt individually (288 s) after all
  batches. At 8/4 cores it passed in its batch; at 128 it was the last component in series 1.
  TFC 16: the IODMA's attempts are sequential (1 parallel attempt at 16 slots).

Open (next improvements, not done):
1. Speculative retry in GenerateBatchPblocks: when a batch runs past ~1.3x its expected time
   and slots are free, start the lower-utilization retry of its items in parallel instead of
   waiting for the limit; start the individual fallback of failed items right away (in
   `pblock_group`), not after all batches.
2. Build small routing-dense MVAUs (or anything that failed a batch before, keyed by
   component hash) individually.
3. At <= 16 slots allow 2 parallel attempts for the few large components.
4. Serial tail: assembly ~150 s is now ~25 % of the CNV total at 64 cores.

Recommended server for these models: 64 physical cores, >= 384 GB (CNV peaks at 273 GB at
64 cores, 385 GB at 128 with 83 slots); more threads (SMT) or cores do not help CNV/TFC.
Larger models (more components) will use more parallelism; memory, ~4.6 GB per Vivado slot
+ ~3 GB per JVM, then bounds it.
