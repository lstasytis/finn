# Issue 013 report: node latency in the analytical FIFO sizer — fixed
(Saved by the orchestrator from the agent's hand-back; subagents cannot write report files.)

## Result (transformer_benchmark ZCU104, fix/013-int = fix/010 + fix/013)
| test | shortcut FIFO_9 | rtlsim / est | FIFO kbit |
|---|---|---|---|
| folding[maximize], before (fix010 v0z1pmrn) | 1232 | 0.73 (1755 cyc) | 210.3 |
| folding[maximize], after (fx/fix013-int/fold-max_..._2u5a4oc_) | 2004 (needs 1501-1550) | 1.00 (1280), PASSED | 246.5 |
| fifo_sizing[heuristic_analytical] (fifo-..._46hghnz1) | 2140 (needs 2065-2077) | 1.00, PASSED | 91.7 (was 89.7) |
| folding[target] (fold-target_..._ykwhj478) | 2140 | 1.00, PASSED | 91.7 (was 89.7) |
Logs: `/home/lstasytis/finn/build/finn_build/fx/fix013-int/logs/tb_*.log`

## Fix (branch `fix/013`, base `feature/dev/analytical-fifo-sizing` c74d17dc5)
* 370225908: every characterized node's writes are delayed by `TAV_PIPELINE_LATENCY` = 32 cycles plus a new hook
  `get_tav_extra_latency()` (InnerShuffle: one page); a new hook `get_tav_input_blocks()` holds a dynamic MVAU's
  outputs until its in1 matrix is loaded (held words released at most 1 per cycle after the block completes).
  `DeriveFIFOSizes` takes the max of 4 passes (causal x latency), so no edge gets smaller than today.
* 0bbdf2fa3: docs, new "Node latency" rule in `fifo_sizing_heuristics.rst`.
Design notes: a delay only matters where paths of different latency meet (joins, shortcuts, the two inputs of a
dynamic MVAU). 32 cycles is above the measured first-output latencies (DWC 1, Softmax 7, EWMul 11, Split 16,
Thresholding_rtl 18, MVAU_rtl 4-69); 16 gave 1538, at the edge of the need. Build-up of the 2004 (offline): latency
alone 1769, + dynamic MVAU hold 1799, + InnerShuffle page 2004. Dropped: issue 010's block patch as-is (bursts, 2->127
on InnerShuffle->DWC edges), holds as time shifts (shortcut 22k-30k), DWC/MVAU feature-map blocks (+2..5 tokens only).
The no-latency passes are kept because latency alone dropped one burst-floor edge (mlp chain test) from 4 to 2.

## Total FIFO size (offline re-sizing on the newest fx/fifo, fx/fifo2 TAVs; base = c74d17dc5 sizer)
| model | base kbit | new kbit | factor |
|---|---|---|---|
| bnn-pynq (9), cybersecurity-mlp, gtsrb, kws, mobilenet_v1, vgg10-radioml, espcn | | | 1.000 |
| resnet50 V80 | 2652.9 | 2660.9 | 1.003 |
| resnet18 U55C | 938.9 | 945.9 | 1.007 |
| yolov8 4w4a / 8w8a | 5172.4 / 9048.1 | 5176.0 / 9055.4 | 1.001 |
| bert VCK190 | 6083.9 | 6158.4 | 1.012 |
| transformer_benchmark committed / maximize | 89.4 / 208.7 | 91.4 / 244.9 | 1.02 / 1.17 |
| transformer_vision / language / radioml | 1173.2 / 5686.1 / 330.1 | 1258.3 / 6233.9 / 383.0 | 1.07 / 1.10 / 1.16 |
Vision/language/radioml have no passing rtlsim today (issue 011), so "already passes" is not established for them.
Sizing pass ~2x slower (resnet50 offline 108 s -> 210 s, 4.5 GB).

## Tests
fix/013: test_fifosizing_chained_tav.py 6 passed (residual_mvau_rtl 88 -> 90; new test_chained_tav_residual_latency
262 -> 449; new test_node_latency_and_block_writes); test_fifosizing.py -k "linear and heuristic_analytical" (tfc,
cnv) 2 passed. fix/013-int: the 3 feature tests above PASSED.

## Open points
OuterShuffle, SWG and FINNLoop have no latency hooks yet; `TAV_PIPELINE_LATENCY` is a module constant (lower
per-op values would cut the over-estimate if the 1.5x limit ever binds); issue 010's aside (fork fed directly by the
graph input is skipped) unchanged.

## Files and deletions
Scripts in scripts/: tav_model.py, size_offline.py, diff_sizes.py, gen_tavs.sh, compare_all.sh, run_tb_tests.sh,
edge_times.py. Data: fx/fix013-int/tav, fx/fix013-int/cmp, fx/fix013/tav. Deleted own dirs only: fx/fix013-int
code_gen/rtlsim/vivado/dataflow_partition (2.3 GB), fx/fix013 code_gen/rtlsim/vivado/build_fifosizing (1.6 GB).
