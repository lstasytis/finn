# Task manual: FIFO-aware folding (follow-up to your optimizer fixes)

You already fixed the folding optimizer (`feature/dev/set-folding-optimizer-fixes`; your
commits were moved onto `feature/dev/set-folding-optimizer`). This is a new, separate task
on the same code. The toolchain is now **Vivado/Vitis 2024.2** (your env script sets it).

## Findings that start this task (measured by the coordinator)

1. **The optimizer's foldings need several times the FIFO storage of the naive folder's**
   at the same throughput target (committed folding JSON's cycles, ZCU104):

   | model | sizing | naive folding | optimizer folding |
   |---|---|---|---|
   | cnv-w1a1 | `heuristic_analytical` (quick mode) | 14.8 kbit | 36.1 kbit |
   | cnv-w1a1 | `largefifo_rtlsim` (full flow, rtlsim) | 26.9 kbit | 165.8 kbit |

   Same rtlsim throughput (12255 vs 12337 fps). FIFOs cost LUTRAM/BRAM, so a folding that
   saves compute but needs 6x the FIFOs may not be cheaper overall. Find out why (which
   edges grow, and what about the optimizer's choice causes it: rate mismatch between
   neighbours? bursty SWG/MVAU pairings? DWCs?).
2. **`enable_folding_fifo_heuristic=True` crashes on dev**: `insert_and_size_fifos` in
   `set_folding.py` builds a `DataflowBuildConfig` with `split_large_fifos` (removed from dev
   in f9fc4be9d) and probes `TAVGenerationMethod`/`TAVUtilizationMethod`, which no longer
   exist: the analytic sizer is now selected by
   `auto_fifo_strategy="heuristic_analytical"` (`AutoFIFOSizingMethod.HEURISTIC_ANALYTICAL`,
   only on trees that carry the FIFO sizing branch).

## Goals

1. Repair the FIFO-aware mode against the current API. It must stay synthesis-free (only
   `heuristic_analytical`, which needs no IP generation; never per-candidate rtlsim/HLS),
   and degrade gracefully (clear warning, no FIFO term) on a tree without the analytic
   sizer: the folding branch is merged into dev **without** the FIFO sizing branch.
2. Make the optimizer's foldings FIFO-economical: at the same target, the FIFO storage of an
   optimizer folding (sized by `heuristic_analytical`) should be close to the naive
   folder's (aim: within 1.2x) unless the compute savings clearly outweigh it; judge by
   total estimated resources including FIFOs. Options: a cheap proxy in the cost model
   (rate/burst mismatch of neighbouring layers), the repaired in-loop analytic sizing (only
   if fast enough), a final FIFO-aware polish, or making the FIFO heuristic the default
   when the analytic sizer exists. Your choice, with measurements.
3. Keep every guarantee you established: feasible targets met, padding never worse
   (`test_padding_tightness` on ZCU104 and U55C), foldings functionally correct, folding
   time reasonable (report before/after).

## Ground rules

* Work only in `/tmp/finn_dev_lstasytis/worktrees/folding-fifo`, branch
  `feature/dev/set-folding-optimizer-fifo-aware` (based on the integration branch: dev +
  folding optimizer + DWC padding + analytic FIFO sizing). `source
  /tmp/finn_dev_lstasytis/agents/folding_fifo/env.sh` in every shell (`FINN_BUILD_DIR` =
  `/tmp/finn_dev_lstasytis/agents/folding_fifo/build`, yours). Do not touch other worktrees,
  the main worktree (`/home/lstasytis/finn_dev_lstasytis/finn`, where a sweep is running) or
  processes you did not start; never push; at most `-n 4`.
* **Disk**: the build volume is 92 GB shared with the sweep and was full twice. HLS projects
  are large; delete the scratch directories your runs create in your own build dir when you
  no longer need them (code_gen_*, rtlsim_*, vivado_stitch_proj_* there), and prefer quick
  mode. Check `df -h /tmp/finn_dev_lstasytis` before full-mode runs.
* Changes only in `src/finn/transformation/fpgadataflow/set_folding.py` (+ folding tests),
  feature-probing anything that exists only with the FIFO sizing branch, so the commits can
  move onto `feature/dev/set-folding-optimizer`. Harness changes, if needed, separately.
* Commit trailer:

  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012VzVo5sDYGjodUk86cC2af
  ```

## Tools

* Quick FIFO-storage comparison (seconds per case):
  `/tmp/claude-1205/-home-lstasytis-finn-dev-lstasytis-finn/8871d419-69de-49fc-824c-30ba6c67e801/scratchpad/fold_fifo_cmp.py`
  (copy and adapt it; it runs the harness's `run_flow` in quick mode with
  `auto_fifo_strategy="heuristic_analytical"` for naive / optimizer / optimizer+fifo on
  cnv-w1a1, cnv-w2a2, tfc-w2a2). The sized model is
  `<output_dir>/intermediate_models/step_set_fifo_depths.onnx`.
* `tests/benchmark/test_dwc_padding_flow.py::test_padding_tightness`,
  `tests/benchmark/test_folding_flow.py` (quick: `FEATURE_FLOW_QUICK=1`),
  `tests/fpgadataflow/test_set_folding.py`.
* FIFO sizing docs (how the analytic sizer reasons, useful to explain the FIFO growth):
  `docs/finn/fifo_sizing_heuristics.rst`.
* Final check in full mode (rtlsim) on cnv-w1a1 only: optimizer folding with
  `largefifo_rtlsim` and `heuristic_analytical` sizing, throughput unchanged.
* Lint: the venv in the previous manual (`.../scratchpad/lintenv`).

## Deliverables

Commits on `feature/dev/set-folding-optimizer-fifo-aware`;
`/tmp/finn_dev_lstasytis/agents/folding_fifo/{PLAN.md,REPORT.md}` (write the plan first;
report: diagnosis of the FIFO growth with edge examples, fix, before/after tables of FIFO
kbit + LUT/BRAM + cycles + folding time per model, the guarantees re-checked, open
questions). Final reply: a short summary with the commit list.
