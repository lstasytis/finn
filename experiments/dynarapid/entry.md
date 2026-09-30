You are continuing work on the FINN + DynaRapid integration, branch `feature/dynarapid-pnr`,
possibly on a different machine than the one that wrote most of the notes.

1. Read CLAUDE.md (repo root) fully — it is the onboarding doc: goal, environment,
   file/function map, commands, known pitfalls, status. Do not re-derive what it states.
   Its "Environment" section describes the original 32-core / 125 GB container; paths, cores
   and memory may differ on your machine (the 128-thread server's layout is in the Status
   section, "Phase 4b").
2. Then read experiments/dynarapid/TASK_scaling_and_examples.md — that is your task.
   Phases 0-4 are done on both machines (results in NOTES.md and NOTES_SCALING.md). Start at
   **Phase 6** (U55C per-model v++ link, section "NEXT AGENT STARTS HERE"), following the
   ground rules at the top of the file; Phase 5 (finn-examples models) follows on the U55C
   afterwards.
3. Use experiments/dynarapid/NOTES.md (and NOTES_SCALING.md for scaling) only as a reference
   for specific past results or decisions (grep for the section you need; don't read them end
   to end).

Preflight on a new machine (before any Phase 6 work):
- `git pull` this branch; `git status` clean apart from local scratch files.
- Toolchain: Vivado + Vitis 2024.2 (`which v++`), U55C platform
  `xilinx_u55c_gen3x16_xdma_3_202210_1` under `$PLATFORM_REPO_PATHS` (default
  `/opt/xilinx/platforms`, mounted by run-docker.sh). Without the platform the U55C work
  cannot run — report that instead of working around it.
- DynaRapid Java must match `docker/dynarapid/*.patch` (this branch added the GreedyPlacer
  recentering needed on the xcu55c). `fetch-repos.sh` cannot apply a newer patch on top of an
  older one. First make sure `deps/DynaRapid` holds no Java changes missing from the patches
  (compare `git -C deps/DynaRapid diff -- . ':!RapidWright'` with the patch file; if they
  differ, stop and ask). Then reset and re-apply:
  `git -C deps/DynaRapid checkout -- . && git -C deps/DynaRapid/RapidWright checkout -- .`,
  `git -C deps/DynaRapid/RapidWright apply $PWD/docker/dynarapid/rapidwright-finn.patch`,
  `git -C deps/DynaRapid apply $PWD/docker/dynarapid/dynarapid-finn.patch` (from the repo
  root; `git -C` resolves relative patch paths inside the submodule), then compile
  (CLAUDE.md "DynaRapid patches").
- The Phase 6 artifacts listed in the task file (VGG10 U55C frontend, Vitis baseline model,
  component library, `accel_routed.dcp`, IP cache) exist only in the original container's
  `$FINN_BUILD_DIR` (`/tmp/finn_dev_lstasytis`). The ONNX models reference generated IP by
  absolute path all over that directory, so they cannot be copied piecemeal. On another machine,
  regenerate them: `run_vgg10.py --board U55C --mode frontend`, then `--mode vivado` (the Vitis
  baseline, 2 h 39 min at 32 cores here; it yields the IODMA `.xo`s and
  `intermediate_models/step_synthesize_bitfile.onnx`, and gives this machine's baseline time).
  Use `$FINN_BUILD_DIR/vgg10_u55c/run_base.sh` there as the template, with your paths; the
  VGG10 model comes from the finn-examples radioml zip (CLAUDE.md "Models"). Then build the
  library and the stitched kernel with `build_library` + `dynarapid_pnr`
  (`place_region=DR_REGION[part]["place"]`), not with the current `dynarapid_alveo_build`
  (it still also builds the obsolete cached shell, ~65 min).

Working style:
- Read only the functions you need (CLAUDE.md gives file:line locations).
- Long builds: run in the background (screen/tmux, or `setsid nohup` where there is none),
  check back instead of waiting, and never run two timed builds at once. Before measuring,
  confirm no other session/container is using `$FINN_BUILD_DIR` or the repo.
- Correctness first: a DynaRapid result counts only with 0 routing errors and a matching
  functional check (verify_accel.py, or the stream-only variant from Phase 6 step 4).
- Record every result in a new dated section of NOTES.md as it arrives, keep the Status
  section of CLAUDE.md current, and commit after each phase. Other nodes push to this branch
  too: pull before you start and merge (not rebase) before pushing.
- If a phase stays blocked after ~2 h of debugging, document your findings and move to the
  next independent phase.
- Alveo U55C work for Phase 6 and Phase 5 is authorized (user decisions 2026-09-29/30). Stop and
  ask me before: starting ResNet50, anything that runs longer than ~8 h, or changing the
  experiment design in the task file.

When done (or blocked), report: the U55C results (VGG10 DynaRapid cold/warm vs. the Vitis
baseline, stage breakdown, WNS, routing errors, functional check), per-model results for the
finn-examples models, whether `run_u55c_vgg10.sh` is ready for the 128-core server, and open
issues.
