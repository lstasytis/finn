You are continuing work on the FINN + DynaRapid integration in
/home/lstasytis/finn_dev_lstasytis/finn (branch feature/dynarapid-pnr).

1. Read CLAUDE.md (repo root) fully — it is the onboarding doc: goal, environment,
   file/function map, commands, known pitfalls, status. Do not re-derive what it states.
2. Then read experiments/dynarapid/TASK_scaling_and_examples.md — that is your task.
   Execute its phases in order (0 → 5), following its ground rules.
3. Use experiments/dynarapid/NOTES.md only as a reference for specific past results or
   decisions (grep for the section you need; don't read it end to end).

Working style:
- Read only the functions you need (CLAUDE.md gives file:line locations).
- Long builds: run in the background, check back instead of waiting, and never run two
  timed builds at once. Before measuring, confirm no other session/container is using
  $FINN_BUILD_DIR or the repo.
- Correctness first: a DynaRapid result counts only with 0 routing errors and a matching
  verify_accel.py check.
- Record every result in a new dated section of NOTES.md as it arrives, keep the Status
  section of CLAUDE.md current, and commit after each phase.
- If a phase stays blocked after ~2 h of debugging, document your findings and move to the
  next independent phase.
- Stop and ask me before: starting ResNet50 or any Alveo work, anything that runs longer
  than ~8 h, or changing the experiment design in the task file.

When done (or blocked), report: the scaling table/plot and what limits scaling, a
recommended server size, per-model results, and open issues.
