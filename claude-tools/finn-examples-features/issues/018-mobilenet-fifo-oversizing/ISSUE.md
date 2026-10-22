# Issue 018: analytical FIFO sizing over-sizes MobileNetV1 (97 kB; LiveFIFO's best ~22 kB)

Read first: `/home/lstasytis/finn/claude-tools/finn-examples-features/GENERAL.md`, `docs/finn/fifo_sizing_heuristics.rst`
on `feature/dev/analytical-fifo-sizing` (the rules), and the reports of issues 006, 013, 015 (rules added recently;
their scripts `perf_override.py` / `size_offline.py` / `compare_all.sh` re-size offline and re-simulate with chosen
FIFO depths) under `/home/lstasytis/finn/claude-tools/finn-examples-features/issues/`.

Data: `test_mobilenetv1_fifo_sizing[U55C-heuristic_analytical-full]` (committed folding, 788544 cycles/frame) passes at
rtlsim = estimate with 97.2 kB of FIFOs (141 FIFOs), build
`/home/lstasytis/finn/build/finn_build/fx/fifo2/fifo-heuristic_analytical_mobilenet_v1_U55C_anmrhapg/`
(`intermediate_models/step_set_fifo_depths.onnx`); the small variant (`...-small`) also passes. Colleagues' LiveFIFO
(per-FIFO minimal search) reaches ~22.1 kB for MobileNetV1 at 394272 cycles (a 2x faster folding than ours, so not the
same optimum; their FINN rtlsim heuristic needs 8.6x, their analytical tree model 4.9x of it). Our 15 largest FIFOs
(76 % of the total) are all on very wide streams: VVAU_hls outputs (640/352/176 bit) into a DWC, depth 80-223, and
FMPadding_rtl outputs (256/128 bit) into the SWG, depth 202-256 (3-8 kB each).

User goal: total FIFO size within 1.5x of the optimum without losing throughput (over-estimating is acceptable
within that bound, under-estimating never).

Tasks:
1. For the large edges, which rule sets the depth (burst floor, refill floor, pacer chains, node latency, phase
   lead, DWC double FIFO, IO, ...)? Instrument the sizer offline (no code change needed for that) and report per edge.
2. Measure the real need: the smallest depth per large FIFO that keeps the throughput (others unchanged, then
   all reduced together), with stitched-IP rtlsim of the small folding first (faster), then the full one. That gives
   an estimate of the optimum for our folding.
3. Fix the rules that over-size (e.g. a backlog counted in wide words where the data volume is smaller, both sides of
   a DWC sized for the same backlog, a margin that scales with stream width) on a branch `fix/018` off
   `feature/dev/analytical-fifo-sizing`, with unit tests and docs. Verify: mobilenet small + full FIFO feature tests
   still pass at rtlsim = estimate; report the total size; re-size the other models offline (`compare_all.sh`) and
   make sure none loses throughput (spot-check in rtlsim any model whose depths shrink).

Rules: at most 16 threads; start nothing while MemAvailable < 200 GB; kill only recorded PIDs (never `pkill -f`/`pgrep
-f` or a `ps | grep` pattern); no RTL/HLS changes; never `largefifo_rtlsim` as a strategy (enlarging chosen FIFOs in a
diagnostic re-simulation is fine); no pushes; don't touch other worktrees, build dirs or the campaign; list deletions.
Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Write REPORT.md next to this file
(if refused, put its full text into your final answer after a <= 10-line summary).
