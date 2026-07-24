# claude-tools

Scratch tooling + working context for the FINN contributions worked on with Claude.
This branch is **never part of any PR** — it is merged into a working branch only to
give the agent (and you) the analysis scripts and the topic context, then the feature
PR is cut from source files only.

## START HERE (status + how to continue) — 2026-07-22

Everything must be **up to date with upstream/dev**. The integration model is:

```
dev-expanded-finnexamples   = upstream/feature/add_finnexamples + merge upstream/dev
                              + benchmark test fixes           (the finn-examples baseline)
dev-<feature>               = dev-expanded-finnexamples + cherry-pick <feature> commit(s)
                              onto dev                          (the working branch)
claude-tools (this branch)  = tooling only; MERGED into working branches transiently,
                              NEVER committed onto examples/PR branches
```

The agent has **no ssh** — it prepares branches locally and prints `git push` commands for
the human to run. `origin = github.com/lstasytis/finn` (fork), `upstream = Xilinx/finn`.

**Done:**
- **Baseline** `dev-expanded-finnexamples` (on latest dev, 0 behind): 5/7 finn-examples
  families build through FINN steps (cybersecurity, gtsrb, vgg10, mobilenet_v1, bnn-pynq
  cnv+tfc). kws xfailed, resnet50 deferred (both need core/streamline changes — out of
  scope). Benchmark dev-drift fixed (`vitis_default_platform` rename, path f-strings,
  resnet50 model name/`step_make_driver`). Models: `tests/benchmark/models/download_models.sh`.
- **Folding** `dev-folding` / `folding-pr` (= dev + optimizer + the **`parallel_window`
  search-space fix**): optimizer now matches the reference-JSON throughput; measured at
  **real OOC synth** by [`folding/folding_eval.py`](folding/CONTEXT.md). See
  [`folding/CONTEXT.md`](folding/CONTEXT.md).

**Next: DWC and FIFO-sizing benchmark testing** — replicate the folding pattern:
1. `git checkout -b dev-<feature> dev-expanded-finnexamples` then
   `git cherry-pick <feature-commit>` (e.g. `origin/feature/generalized-datawidthconverter`
   or `origin/feature/analytical-fifo-sizing`); resolve any dev-phase-refactor conflicts.
2. Write a cached, **synthesis-based** eval harness in this folder, modeled on
   `folding/folding_eval.py`: cache the shared checkpoint, re-run only the transform under
   test, read resources from the OOC **synth** report (regular synth, no P&R — set
   `stitched_ip_gen_dcp=True`, parse `<proj>/stitched_ip/finn_design_partition_util.rpt`;
   top row `| finn_design_wrapper ` = LUT/LogicLUT/LUTRAM/SRL/FF/RAMB36/RAMB18/DSP).
3. DWC: compare generalized HLS DWC **LUT** vs default rtl/hls per model — see
   [`dwc/CONTEXT.md`](dwc/CONTEXT.md). FIFO: compare analytic tree-model KB vs
   analytic_rtlsim/largefifo_rtlsim vs paper ground truth — see
   [`fifo-sizing/CONTEXT.md`](fifo-sizing/CONTEXT.md). The old cached gtsrb fifo-variant
   benchmark test (search build dir, re-run `step_set_fifo_depths` from a cached
   `step_hw_ipgen`) needs re-integrating on the new dev tree.

**Driver gotchas:** `build_dataflow_cfg` returns `-1` on failure (doesn't raise) and drops
into pdb unless `cfg.enable_build_pdb_debug=False`. dev refactored build steps into 6
**phases** (`build_dataflow_phases.py`); truncate step lists by phase name. To run a build
past the pytest vivado-version guards, drive `build.build_dataflow_cfg` directly (see the
generic driver in the folding eval).

## Workflow

1. You work on one of three feature areas (see subfolders below). Each has a
   `CONTEXT.md` describing the goal, the branches, the key source files, current
   status, and how to run its tooling.
2. Merge (or cherry-pick) this `claude-tools` branch into your working branch so the
   `claude-tools/` directory is present in the tree while you work.
3. Keep `claude-tools/` **out of the feature commit** — PRs are assembled from the
   actual `src/` + `tests/` changes only, squashed into a single clean, lint-passing
   commit.

## Topics

| Folder | Feature branch | What it is |
|---|---|---|
| [`dwc/`](dwc/CONTEXT.md) | `feature/generalized-datawidthconverter` | Generalized StreamingDataWidthConverter (padding/cropping + arbitrary widths) |
| [`folding/`](folding/CONTEXT.md) | `feature/set-folding-optimizer` | Resource-aware folding optimizer (cost fns + simulated annealing) |
| [`fifo-sizing/`](fifo-sizing/CONTEXT.md) | `feature/analytical-fifo-sizing` | Analytical (tree-model) FIFO sizing |

These three are related: folding sets stream widths → drives DWC insertion → drives
FIFO depths. The DWC is a dependency for the folding optimizer to work well.

## Conventions (repeat of the repo CLAUDE.md rules)

- **Minimal code changes**; match surrounding style; **minimal comments** (1–2 lines).
- **Never push** to the reviewed feature branches without being asked; PRs are cut as
  a **single squashed commit** and must pass FINN lint (black/isort/flake8, line 100).
- Test via pytest; we mainly care about **finn-examples** and the per-op fpgadataflow
  tests. rtlsim often can't run locally (verilator 5.x vs old pyverilator gap) — use
  cppsim where possible.

## Linting (what CI enforces)

Pinned versions: `black==23.3.0` (`--line-length=100`), `isort==5.12.0`,
`flake8==6.0.0` (`--max-line-length=100 --extend-ignore=E203`). See
`.pre-commit-config.yaml`. These are **not** in the system python — a persistent,
gitignored lint env is kept at `.lintenv/` with the exact run commands: see
[`LINTING.md`](LINTING.md). Run it before every commit that touches `.py`.
