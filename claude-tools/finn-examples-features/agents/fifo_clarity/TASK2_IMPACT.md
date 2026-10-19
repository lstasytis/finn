# Follow-up task: rank the FIFO sizing heuristics by measured impact

Your earlier work (refactor, docs, regression test) is integrated in
`feature/dev/analytical-fifo-sizing`; your worktree was fast-forwarded to it (afa1586b0:
adds a word-based DWC cycle estimate, no depth change on any fixture, and a harness that
prunes HLS synthesis databases). The toolchain is now **Vivado/Vitis 2024.2**:
`/tmp/finn_dev_lstasytis/agents/fifo_clarity/env.sh` was rewritten for it and the XSI
library rebuilt. Earlier rtlsim numbers were on 2023.1; re-measure everything you report.

## What the user asked (their words, condensed)

> Rank the heuristics by their impact %-wise for all the models overall: which are the
> heavy-duty ones and which are minor improvements / correcting small throughput losses.
> Distinguish heuristics that **minimize FIFO depth** from those that **help maintain high
> throughput** (if a minimizing heuristic over-minimizes, the throughput-focused one
> corrects for that). For each model, a table of the total % buffer depth change and the
> throughput change once a heuristic is applied.

"Applied" = the full sizer vs the sizer with only that rule removed (your ablations), so
report `depth(with) vs depth(without)` and `fps(with) vs fps(without)` per rule and model.

## Do

1. Depth: you already have the ablation totals for 9 models
   (`/tmp/claude-1205/-home-lstasytis-finn-dev-lstasytis-finn/8871d419-69de-49fc-824c-30ba6c67e801/scratchpad/ablation_summary.txt`)
   and resnet50 (`fixtures/resnet50/abl*_*.json`). Report both total depth and total
   storage (kbit, depth x stream width), since widths differ wildly between edges.
2. Throughput: stitched-IP rtlsim (ZCU104, 1 frame, your `throughput.sh` flow, fixed depths
   via `auto_fifo_depths=False`) of the full sizer and of every ablation that changes a
   depth, on cnv-w1a1, cnv-w2a2, gtsrb, kws, vgg10-radioml. tfc-w1a1/tfc-w2a2/cybersecurity
   are unaffected by every rule (identical depths), say so. Try mobilenet_v1 on 2024.2
   (the URAM-init problem was seen on 2023.1); resnet50 only if it is practical (V80, huge
   HLS), otherwise depth-only and say so. Also run the `largefifo_rtlsim` reference per model
   if you do not have it on 2024.2, so each table has the reference fps.
3. Classify every rule as **depth-minimizing** (removing it makes FIFOs deeper) or
   **throughput-protecting** (removing it makes FIFOs shallower and risks throughput), and
   for each protecting rule name the minimizing rule(s) it corrects (e.g. the burst floor
   restores what slack relaxation / paced input take away). Rules that change nothing on
   any model: say so (they are candidates for removal; the user allows removing or
   simplifying a heuristic if FIFO depth grows by at most a few % and throughput never
   drops; if you find such a case, propose it, do not implement it).
4. Rank by overall impact: geometric-mean / median % depth change across models plus the
   worst-case throughput loss when removed. Heavy-duty vs minor.

## Deliver

* A new section "Impact of each rule" in `docs/finn/fifo_sizing_heuristics.rst`: the ranking
  table (rule, class, median/max depth effect, worst throughput effect, models it matters
  on) and one compact per-model table (rows = rules, columns = depth %, storage %, fps %).
  Commit it on your branch (`feature/dev/analytical-fifo-sizing-clarity`; nothing pushed).
* `/tmp/finn_dev_lstasytis/agents/fifo_clarity/IMPACT.md` with the same tables and raw
  numbers, and a short final reply with the ranking in a few lines.

## Constraints (unchanged, plus disk)

* Only your worktree and build dir; the main worktree runs a sweep; another agent may run;
  at most `-n 4` / 4 parallel runs.
* **Disk**: the build volume (92 GB) is shared and filled up twice. Delete your runs'
  `code_gen_*`, `rtlsim_*`, `vivado_stitch_proj_*` scratch in your build dir as soon as a
  variant's numbers are recorded (keep what the IP cache index references, minus
  `sol1/.autopilot`); check `df -h /tmp/finn_dev_lstasytis` before big runs.
* Commit trailer as before (Claude Fable 5.1 + the Claude-Session line).
