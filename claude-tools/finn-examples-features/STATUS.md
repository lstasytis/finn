# Feature-test status (models x features)

Generated 2026-10-19 13:49 by `tools/status_table.py` from the campaign results. Cells: result, rtlsim/estimate throughput ratio, (folding/DWC) own estimate vs the committed folding's estimate, FIFO storage, padding DWCs, time of the feature's step (FIFO sizing or folding; measured under load, not comparable to isolated timings). ⏳ running, ⏸ not run yet, aborted = killed at the reboot deadline.

| model (board) | est. fps (committed) | FIFO analytical | Folding target | Folding maximize | DWC | FIFO rtlsim | Notes |
|---|---|---|---|---|---|---|---|
| tfc-w1a1 (AUP-ZU3_8GB) | 11111111.1 | ⏳ | ⏳ | ⏸ | ⏸ | ⏸ | folding target: committed folding near-fully unrolled (9 cycles) - beyond the optimizer's ZU3 budget and LabelSelect kept at PE=1; optimizer 15 cycles, rtlsim = estimate |
| tfc-w1a2 (AUP-ZU3_8GB) | 6666666.7 | ✅ 1.000x 8.2 kbit 0.2 s | ⏳ | ⏸ | ⏸ | ⏸ |  |
| tfc-w2a2 (AUP-ZU3_8GB) | 6666666.7 | ✅ 1.000x 6.7 kbit 0.2 s | ⏳ | ⏸ | ⏸ | ⏸ |  |
| lfc-w1a1 (AUP-ZU3_8GB) | 192678.2 | ✅ 0.988x 4.8 kbit 0.6 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| lfc-w1a2 (AUP-ZU3_8GB) | 48661.8 | ✅ 0.997x 6.8 kbit 0.6 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w1a1 (AUP-ZU3_8GB) | 12244.4 | ✅ 1.001x 11.9 kbit 1.9 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w1a2 (AUP-ZU3_8GB) | 3028.5 | ✅ 1.008x 26.7 kbit 2.9 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w2a2 (AUP-ZU3_8GB) | 3028.5 | ✅ 1.008x 25.5 kbit 3.2 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w1a2 (KV260_SOM) | 3028.5 | ✅ 1.008x 26.7 kbit 2.8 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w2a2 (U55C) | 3028.5 | ✅ 1.007x 25.5 kbit 3.2 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| gtsrb (AUP-ZU3_8GB) | 8133.4 | ✅ 1.000x 9.9 kbit 2.3 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| kws (AUP-ZU3_8GB) | 25464.7 | ✅ 0.998x 4.3 kbit 0.1 s | ✅ 0.998x est 1.00x ref 4.8 kbit 1.9 s | ✅ 0.988x est 8.00x ref 5.5 kbit 26.8 s | skip (no padding) | ⏸ |  |
| cybersecurity-mlp (AUP-ZU3_8GB) | 12500000.0 | ✅ 1.333x 3.7 kbit 0.1 s | ⏳ | ✅ 1.000x est 0.35x ref 3.5 kbit 17.7 s | skip (no padding) | ⏸ |  |
| vgg10-radioml (ZCU104) | 243190.7 | ✅ 1.001x 14.2 kbit 0.4 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| espcn (KV260_SOM) | 381.5 | ⏳ | ⏳ | ⏳ | skip (no padding) | ⏸ |  |
| mobilenet_v1 (U55C) | 422.7 | ✅ 1.000x 777.5 kbit 47.1 s | ⏳ | ⏳ | ⏳ | ⏸ |  |
| resnet18_4w4a (U55C) | 3814.7 | ⏳ | ❌ 0.2 s #003 | ❌ 0.1 s #003 | ❌ #003 | ⏸ |  |
| resnet18_8w8a (U55C) | 1695.4 | skip | ❌ 0.1 s #003 | ❌ 0.2 s #003 | ❌ #003 | ⏸ | FIFO: n/a, committed folding has runtime-writeable URAM weights, not initialized in stitched-IP rtlsim |
| quantyolov8_4w4a (U55C) | 90.4 | ⏳ | ❌ 0.3 s #003 | ❌ 0.2 s #003 | ⏳ | ⏸ |  |
| quantyolov8_8w8a (U55C) | 90.4 | ⏳ | ❌ 0.3 s #003 | ❌ 0.3 s #003 | ⏸ | ⏸ |  |
| bert (VCK190) | 508.6 | ✅ 4.000x 6085.0 kbit 20.1 s | ❌ 0.1 s #003 | ❌ 0.1 s #003 | ⏸ | ⏸ | FIFO: #001 fixed (dynamic MVAU characterization), rerun; estimate 4x pessimistic for tiled MVAUs (TH), rtlsim 4.0x est |
| resnet50 (V80) | 276.8 | ⏳ | ❌ 0.4 s #003 | ❌ 0.4 s #003 | ⏸ | ⏸ |  |

Issues: `claude-tools/finn-examples-features/issues/<NNN-...>/ISSUE.md` (+ `REPORT.md` of the fix agent).
* #001 fixed: bert dynamic MVAUs now characterized (tree model + in1 TAV); landed on feature/dev/analytical-fifo-sizing.
* #002 fixed: MVAU_hls/LabelSelect_hls per-frame HLS overhead (+7/+4 cycles) added to get_exp_cycles on the new standalone branch feature/dev/hls-exp-cycles-overhead; tfc/cybersecurity jobs rerun with it.
* #003 folding optimizer raises on branching graphs (SWG consumer = next node) for resnet18/50 and yolov8, and on bert (dynamic MVAU without weight initializer).
* DWC tests: at the reference target the seeded optimizer inserts no padding DWC on most models (skip = no padding needed).
* #004 fixed: optimizer repair loop missed unnamed DWC bottlenecks (vgg10 1280 -> 1028 cycles); no padding next to a reshape (InsertDWC assert).
* #005 fixed: phantom URAM estimate of embedded MVAU weights (dev bug) + maximize search fixes; maximize now fits the budget and matches/beats the committed folding where that fits.
* #006 open (agent running): balanced optimizer foldings on cnv: rtlsim 0.95-0.98x of the estimate.
* Folding/DWC jobs of bnn-pynq, gtsrb, vgg10 rerun after fixes 004/005 (13:47).
