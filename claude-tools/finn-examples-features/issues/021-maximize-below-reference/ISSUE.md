# Issue 021: maximize-mode folding below the committed folding's throughput

## Symptom
`test_<model>_folding[...-maximize-full]` (`folding_style="optimizer"`, no target, no folding JSON,
`folding_seed=1`): the estimated throughput of the maximized folding relative to the committed
folding's estimate ("ref", STATUS.md, campaign variant 2, 2026-10-21):

| model | board | maximize / ref |
|---|---|---|
| tfc-w1a1 | AUP-ZU3_8GB | 0.39x |
| tfc-w1a2 | AUP-ZU3_8GB | 0.43x |
| tfc-w2a2 | AUP-ZU3_8GB | 0.43x |
| cnv-w1a1 | AUP-ZU3_8GB | 0.57x |
| cybersecurity-mlp | AUP-ZU3_8GB | 0.35x |
| gtsrb | AUP-ZU3_8GB | 0.96x |
| espcn | KV260_SOM | 0.67x |

Others met or beat the reference (cnv-w1a2 1.19x, kws 8x, lfc-w1a2 2x, bert 1.5x); mobilenet
(15.7x, U55C) fails its rtlsim/estimate check at 0.88x (not part of this issue).

## Earlier diagnosis (to verify)
On ZU3 maximize is limited by FINN's pessimistic LUT estimate for binary/low-bit HLS MVAUs: the
committed folding's estimate already exceeds the budget the optimizer allows (70 % of the LUTs),
yet that folding is implemented on the real device.

## Task
1. Per model: the binding resource, the budget fraction used, the committed folding's estimate vs
   the budget; search problem or budget/estimate problem.
2. Real resource use vs FINN's estimates per node type: post-synthesis utilization of existing
   builds, else out-of-context synthesis of small designs (tfc, cnv, cybersecurity on ZU3).
3. Improve: relax the maximize utilization limit (justified margin) and/or calibrate the
   estimates (HLS/RTL MVAU, VVAU, SWG, Thresholding LUT/BRAM). Estimate changes on a separate
   PR branch `feature/dev/<name>` off the dev base of the other estimate branches; optimizer and
   budget changes on `feature/dev/set-folding-optimizer` (another agent adds an HLS stream-width
   limit there concurrently: pull/rebase, keep separate). No RTL/HLS source changes.
4. Validate: land on the test2 branches, rerun the affected `v2__fold-max__*` jobs through the
   campaign runner (`tools/requeue.sh`). A maximize result counts only if it passes its
   rtlsim/estimate check (>= 0.98) and fits the board; with a relaxed limit show for at least
   one synthesized design per board that the larger folding still fits.

## Rules
At most 16 threads, nothing new below 200 GB MemAvailable, kill only own PIDs (never `pkill -f`),
no `largefifo_rtlsim`, no pushes. Details in `REPORT.md` next to this file.
