# 025: throughput failures on the FIFO PR head after the dev sync (447b96e25)

Verdict: **not the FIFO sizer.** With every StreamingFIFO at depth 64 the steady-state interval is identical in
all four models.

## Per model

| model | estimate (cycles/frame) | N=2 | N=10 | N=50 | cause |
|---|---|---|---|---|---|
| cybersecurity-mlp | 1 | 2 | 1.11 | 1.02 | measurement: one-cycle startup slip, N=2 sees only the first gap |
| tfc-w1a1 | 3 | 7 | 7 | 7 | LabelSelect_hls restarts per frame (+4) |
| tfc-w1a2 | 8 (LabelSelect 6) | 9 | 9 | 9 | LabelSelect_hls (+3) |
| tfc-w2a2 | 8 (LabelSelect 6) | 9 | 9 | 9 | LabelSelect_hls (+3) |

Same numbers with the sized FIFOs and with all FIFOs at 64.

* **tfc:** LabelSelect_hls alone (N=20) measures 7 / 9 / 9 against estimates 3 / 6 / 6, exactly the whole-design
  interval. It is still a per-frame-restart `ap_ctrl_none` top (not made free-running upstream). Before the sync
  the MVAU HLS overhead was larger and hid it. All other nodes match their estimates: the free-running MVAU cores
  (upstream b5c18ada7) run at exactly SF*NF (tfc: 2, 8, 5); Thresholding/DWC estimate + 0.05 (startup slip spread
  over 19 gaps).
* **cybersecurity-mlp:** steady state is 1 cycle/frame = estimate (N=50 average 1.02, last gap 1). MVAU-only
  chains give 1.0 at N=2; input FIFO + MVAU0 gives 2.0. Any `fifo.sv` instance alone (depth 2, 5 SRL, 64
  distributed) gives 2 at N=2, 1.5 at N=3, 1.11 at N=10: depth-independent, `fifo.sv` and the xsi driver are
  identical to upstream/dev; most likely a reset/driver interaction at startup (exact cycle not found).
* **N=2 is misleading for 1-cycle frames:** `stable_throughput` gives 0.5x at N=2, 0.90x at N=10, 0.98x at N=50.
  `fps_from_interval` (last gap) is correct from N>=3. `interval_cycles` is the last gap, not the maximum.

## Suggested fixes (not FIFO sizing)

* `LabelSelect_hls.get_exp_cycles` + 4 (the LabelSelect half of integration commit 74109d25e): estimates 7/10/10
  vs measured 7/9/9, all pass. Own small estimates PR.
* 74109d25e also adds `hls_overhead=7` to MVAU_hls; after the free-running change that over-estimates every
  MVAU-bound model by ~7 cycles/frame: drop it when the integration branch takes the sync.
* Measurement: compare `fps_from_interval` with `rtlsim_batch_size >= 3`, or batch >= 100; or measure
  `stable_throughput` from the 2nd completed frame on in `finn_xsi/rtlsim_xsi.cpp` (rtlsim tooling PR).

## Method

Re-stitched each build's `step_measure_rtlsim_performance.onnx` as built and with all FIFOs at 64, same xsi
throughput measurement (behav=False) at N=2/10/50; each compute node alone (N=20); short node chains (N=2, 10).
Scripts: `scripts/fifo_override_perf.py`, `node_perf.py`, `chain_perf.py`, `run*.sh` (source the removed worktree
`build/wt/devsync-perf` at 447b96e25 with `deps -> build/deps_devsync`; re-create it to rerun; it needed `xsi.so`
copied from pr-fifo). Results/logs: `build/finn_build/fx/devsync_perf/{results,logs}/`.
