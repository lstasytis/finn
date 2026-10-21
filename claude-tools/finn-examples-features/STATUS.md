# Feature-test status (models x features)

Generated 2026-10-21 00:38 by `tools/status_table.py` from the campaign results (variant 2). Cells: result, rtlsim/estimate throughput ratio, (folding/DWC) own estimate vs the committed folding's estimate, FIFO storage, padding DWCs, time of the feature's step (measured under load). With sizes, a cell lists the small folding first, then the full one. ⏳ running, ⏸ not run yet, blocked = its small run failed. FIFO-aware folding (optional, `enable_folding_fifo_heuristic`) at the reference target: folding time, throughput, final FIFO size, each vs the plain optimizer folding.

| model (board) | est. fps (committed) | FIFO analytical | Folding target | Folding maximize | DWC | FIFO-aware: time | FIFO-aware: throughput | FIFO-aware: FIFO size | FIFO rtlsim | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| tfc-w1a1 (AUP-ZU3_8GB) | 11111111.1 | ✅ 1.000x 5.1 kbit 0.1 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏳ / ⏸ | ⏸ |  |  | ⏸ / ⏸ | folding target: committed folding near-fully unrolled (9 cycles) - beyond the optimizer's ZU3 budget and LabelSelect kept at PE=1; optimizer 15 cycles, rtlsim = estimate |
| tfc-w1a2 (AUP-ZU3_8GB) | 6666666.7 | ✅ 1.020x 2.7 kbit 4.0 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| tfc-w2a2 (AUP-ZU3_8GB) | 6666666.7 | ✅ 1.019x 2.4 kbit 3.6 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| lfc-w1a1 (AUP-ZU3_8GB) | 192678.2 | ✅ 1.000x 3.5 kbit 4.5 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| lfc-w1a2 (AUP-ZU3_8GB) | 48661.8 | ✅ 1.000x 5.5 kbit 4.5 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| cnv-w1a1 (AUP-ZU3_8GB) | 12244.4 | ✅ 1.001x 13.9 kbit 2.5 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| cnv-w1a2 (AUP-ZU3_8GB) | 3028.5 | ⏳ / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| cnv-w2a2 (AUP-ZU3_8GB) | 3028.5 | ⏳ / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| cnv-w1a2 (KV260_SOM) | 3028.5 | ⏳ / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| cnv-w2a2 (U55C) | 3028.5 | ⏳ / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| gtsrb (AUP-ZU3_8GB) | 8133.4 | ✅ 1.000x 9.9 kbit 4.3 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| kws (AUP-ZU3_8GB) | 25464.7 | ✅ 1.000x 4.2 kbit 0.2 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| cybersecurity-mlp (AUP-ZU3_8GB) | 12500000.0 | ✅ 1.100x 1.9 kbit 0.1 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| vgg10-radioml (ZCU104) | 243190.7 | ✅ 1.001x 11.0 kbit 13.1 s / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| espcn (KV260_SOM) | 381.5 | ⏳ / ⏸ | ⏳ / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ | #007: output zeros under backpressure (FIFO analytical, fold-target); fold-max passes; OuterShuffle_hls II=3 (upstream) |
| mobilenet_v1 (U55C) | 422.7 | ✅ 0.991x 519.0 kbit 163.4 s / ⏸ | ✅ 0.986x est 1.00x ref 561.2 kbit 22.7 s / ✅ 0.987x est 1.00x ref 87.9 kbit 23.1 s | ❌ 0.883x est 15.71x ref 1132.8 kbit 392.4 s (see log) | ✅ 0.983x est 1.00x ref 545.4 kbit 6 pad DWC 46.7 s / ❌ 0.979x est 1.00x ref 721.5 kbit 23 pad DWC 49.4 s (see log) | ⏸ |  |  | ⏸ / ⏸ |  |
| resnet18_4w4a (U55C) | 3814.7 | ⏳ / ⏸ | ✅ 1.000x est 1.00x ref 1457.0 kbit 29.1 s / ⏳ | ⏳ | ✅ 1.000x est 1.00x ref 1485.9 kbit 2 pad DWC 53.8 s / ⏳ | ⏸ |  |  | ⏸ / ⏸ |  |
| resnet18_8w8a (U55C) | 1695.4 | skip / ⏸ | ⏳ / ⏸ | ⏳ | ⏳ / ⏸ | ⏸ |  |  | ⏸ / ⏸ | FIFO: n/a, committed folding has runtime-writeable URAM weights, not initialized in stitched-IP rtlsim |
| quantyolov8_4w4a (U55C) | 90.4 | ⏳ / ⏸ | ⏳ / ⏸ | ⏳ | ⏳ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| quantyolov8_8w8a (U55C) | 90.4 | ⏳ / ⏸ | ⏳ / ⏸ | ⏳ | ⏳ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| bert (VCK190) | 2034.5 | ❌ 3510.0 kbit 457.1 s (see log) / ✅ 1.000x 6085.0 kbit 20.3 s | ❌ est 1.00x ref 2714.5 kbit 14.0 s (see log) / ⏸ | ⏸ | ⏸ / ⏸ | ⏸ |  |  | ⏸ / ⏸ | FIFO: #001 fixed (dynamic MVAU characterization), rerun; estimate 4x pessimistic for tiled MVAUs (TH), rtlsim 4.0x est |
| resnet50 (V80) | 276.8 | ⏳ / ⏸ | ⏳ / ⏸ | ❌ est 12.00x ref 2289.0 kbit 2232.9 s (see log) | ⏳ / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |
| transformer_benchmark (ZCU104) | 1525.9 | ✅ 1.000x 89.7 kbit 1.0 s / ✅ 1.000x 89.7 kbit 1.1 s | ✅ 1.000x est 1.00x ref 89.7 kbit 6.8 s / ✅ 1.000x est 1.00x ref 89.7 kbit 7.2 s | ✅ 1.000x est 51.20x ref 246.5 kbit 143.5 s | skip (no padding) / skip (no padding) | 8 s (plain 7 s) | ✅ 1525.9 fps (1.000x est) | 89.7 kbit (plain 89.7) | ⏸ / ⏸ |  |
| transformer_radioml (ZCU104) | 21701.4 | ❌ 278.2 kbit 2.9 s (see log) / ❌ 330.2 kbit 1.3 s (see log) | ❌ est 1.00x ref 302.7 kbit 10.7 s (see log) / ❌ 2.667x est 0.75x ref 414.7 kbit 20.9 s (see log) | ❌ 2.667x est 0.75x ref 419.9 kbit 240.9 s (see log) | ⏸ / ⏸ | 275 s (plain 21 s) | ❌ 43402.8 fps (4.000x est) | 284.4 kbit (plain 414.7) | ⏸ / ⏸ |  |
| transformer_vision (ZCU104) | 1525.9 | ❌ 527.3 kbit 13.7 s (see log) / ❌ 1167.6 kbit 11.2 s (see log) | ❌ 0.883x est 1.00x ref 574.0 kbit 27.8 s (see log) / ❌ est 1.00x ref 1207.0 kbit 25.3 s (see log) | ❌ 0.267x est 4.00x ref 1783.5 kbit 390.8 s (see log) | ❌ 0.899x est 1.12x ref 1328.5 kbit 1 pad DWC 46.4 s (see log) / ⏸ | 366 s (plain 25 s) | ❌ 1525.9 fps (1.000x est) | 1230.5 kbit (plain 1207.0) | ⏸ / ⏸ |  |
| transformer_language (VCK190) | 1017.3 | ❌ 5049.5 kbit 85.0 s (see log) / ⏸ | ⏸ / ⏸ | ⏸ | skip (no padding) / ⏸ | ⏸ |  |  | ⏸ / ⏸ |  |

Issues: `claude-tools/finn-examples-features/issues/<NNN-...>/ISSUE.md` (+ `REPORT.md` of the fix agent).
* #001 fixed: bert dynamic MVAUs now characterized (tree model + in1 TAV); landed on feature/dev/analytical-fifo-sizing.
* #002 fixed: MVAU_hls/LabelSelect_hls per-frame HLS overhead (+7/+4 cycles) added to get_exp_cycles on the new standalone branch feature/dev/hls-exp-cycles-overhead; tfc/cybersecurity jobs rerun with it.
* #003 fixed (18:45, landed, big models not rerun before the deadline): SWG paired with its real consumer; streamed (dynamic) MVAU weights not resized/padded; two InsertDWC fixes (DWC branch). Optimizer target mode meets the target on resnet18/50, yolov8, bert; resnet18 4w4a fold-target verified in stitched-IP rtlsim by the fix agent.
* #004 fixed: optimizer repair loop missed unnamed DWC bottlenecks (vgg10 1280 -> 1028 cycles); no padding next to a reshape (InsertDWC assert).
* #005 fixed: phantom URAM estimate of embedded MVAU weights (dev bug) + maximize search fixes; maximize now fits the budget and matches/beats the committed folding where that fits.
* Folding/DWC jobs of bnn-pynq, gtsrb, vgg10 rerun after fixes 004/005 (13:47).
* DWC tests: at the reference targets the padded optimizer (seeds 1-8 tried) inserts no padding DWC on the small models; same with the pre-fix set_folding.py (checked on tfc-w1a1), so padding is not broken, it just doesn't pay off at these targets.
* #006 fixed: FIFO sizing (pacer-bound chains + refill floor) for balanced foldings; cnv maximize now 1.000x. mobilenet maximize: ~80 nodes at exactly the target period, 12 % loss not FIFO-related (optimizer margin?); mobilenet DWC 0.976x is FIFO sizing, edge not found.
* #007 open: espcn stitched-IP rtlsim replaces whole pixels by 0 in a fixed pattern from ~29 % of the frame, only in the full design under backpressure (golden and Python flow correct, all sub-graphs correct); also OuterShuffle_hls II=3 (3x its estimate, upstream).
