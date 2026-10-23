# 023: RapidWright islands flow as an additional benchmark on the finn-examples models

User (2026-10-22): "any free cpu resources within your allowed quota should also be used to now
reintroduce the rapidright-islands flow as an additional benchmarking test" and "I would start with
the small and then full size existing folding models to test".

## Goal

Per finn-examples model: build a **bitstream with the RapidWright island flow** from the
**committed folding**, first the `_small.json` folding, then the full committed one (same order as
the campaign's `--sizes small,full`). Check functional correctness (`verify_accel`), and record the
build time next to FINN's normal Vivado bitfile flow (the baseline) for the same model and folding.
Functional correctness is mandatory; QoR and time are the benchmark numbers.

## Where the flow lives

Branch `feature/rapidwright-islands` (forked from `feature/dynarapid-pnr`, base = upstream/dev of
2026-10-09, last commit 653fa6d6f of 2026-10-19). Read its `CLAUDE.md` (sections "Island flow" and
"Status") first, then `experiments/dynarapid/NOTES_ISLANDS.md` only where needed (1400 lines; last
two sections = current state). Entry points: `ZynqBuild(dynarapid={"flow": "islands", ...})`,
`src/finn/util/rwislands/`, `experiments/dynarapid/run_bitfile_experiment.py --mode islands|vivado`,
`run_islands_timing.sh` (ZCU104), `run_vshell_timing.sh` (U55C Vivado-only shell), `verify_accel.py`.

Trusted configurations only: island flow (grid/skyline islands) and the overlay with the **greedy**
planner and **per-node synthesis** (`FINN_RWI_SYNTH_BATCH=0`, no `plan=dp`): batch synthesis + dp
failed verification (open bisection, paused). Results before FLOW_VERSION 9 are invalid
(zero-threshold bug). 100 MHz unless the model's build config says otherwise.

## Steps

1. Port: new branch `test2/islands` = `dev-finnexamples-features` + merge of
   `feature/rapidwright-islands` (the integration branch carries all models, the per-model feature
   tests and all fixes). Worktree `build/wt/islands2` (`tools/wtsetup.sh`). Resolve conflicts in
   favour of the integration branch for everything that is not island/dynarapid code (DWC, FIFO,
   folding, exp-cycles fixes); keep the DWC invariant in `GENERAL.md`. Unit tests of the islands
   branch's own ops (StreamingLaneSplit/Merge, split_large_mvau) must still pass. Never touch the
   `feature/dev/*` PR branches or the campaign's `test2/*` worktrees.
2. Test: add `test_<m>_islands[small|full]` (marker `islands`) per model file on `test2/islands`
   only, self-contained like the other feature tests: resume the cached reference estimate
   checkpoint (or the committed-folding build), run the island bitfile flow, run `verify_accel`
   against the stitched-IP reference with the model's golden I/O, write
   `report/feature_metrics.json` (islands time + breakdown, WNS/WHS, routing errors, utilisation,
   verify result). A baseline Vivado bitfile build of the same checkpoint is a separate job
   (`islands-baseline`), so the time ratio is known; run it once per model/size.
3. Boards: islands support Zynq (ZCU104 shell) and U55C (Vivado-only shell). AUP-ZU3_8GB / KV260_SOM
   models: build for their own part if the shell builder can be made to support it cheaply,
   otherwise ZCU104 with the same folding (say which in the results). Versal (VCK190, V80): skip,
   no shell (and no VCK190 synthesis licence).
4. Order: all small foldings first (small models first: tfc, lfc, cnv, cybersecurity, kws, gtsrb,
   vgg10, transformer_benchmark, mobilenet, resnet18, espcn, yolov8, transformer_*), then the full
   foldings in the same order. A full job only after its small sibling passed.
5. Results: STATUS-style table in `issues/023-islands-benchmark/RESULTS.md` (model, board, size,
   islands time, baseline time, speedup, timing met, verify) and the details in REPORT.md. Failures
   that are flow bugs: fix on `test2/islands` with a short note per fix; open questions in REPORT.md.

## Resource rules (shared machine, the campaign keeps running)

* The campaign runner (`build/finn_build/fx/logs/campaign/runner.pid`) owns the main budget: it
  launches while the 1-min load is <= 115. The islands jobs only use what is left: start a job only
  while load + its threads <= 115 and `MemAvailable` > 200 GB; size `workers` to the free threads
  (typically 16-32) and never exceed them. Drive the jobs with your own small driver that writes its
  own PID file, gates on load/memory before each start and stops launching (never kills) when the
  campaign needs the room. Run `experiments/dynarapid/mem_watchdog.sh` (islands branch) next to it.
* Timings are measured under the campaign's load: record the load at start/end and the thread cap
  next to every time, and run each islands job and its baseline with the same thread cap.
* Never `pkill -f`/`pgrep -f`; kill PID trees. Leave the paused (state `T`) bisection processes alone.
* `FINN_BUILD_DIR=/home/lstasytis/finn/build/finn_build/fx/islands2`; shells cached in
  `$FINN_BUILD_DIR/rwislands/shells` (copy existing ones from `build/finn_build/rwislands/shells`
  if their key matches). Disk is at 96 %: keep >= 60 GB free, delete island intermediates of
  finished passing jobs (keep reports, logs, bitstream, metrics).
