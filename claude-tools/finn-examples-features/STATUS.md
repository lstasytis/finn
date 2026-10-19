# Feature-test status (models x features)

Generated 2026-10-19 11:45 by `tools/status_table.py` from the campaign results. Cells: result, rtlsim/estimate throughput ratio, (folding/DWC) own estimate vs the committed folding's estimate, FIFO storage, padding DWCs, time of the feature's step (FIFO sizing or folding; measured under load, not comparable to isolated timings). ⏳ running, ⏸ not run yet, aborted = killed at the reboot deadline.

| model (board) | est. fps (committed) | FIFO analytical | Folding target | Folding maximize | DWC | FIFO rtlsim | Notes |
|---|---|---|---|---|---|---|---|
| tfc-w1a1 (AUP-ZU3_8GB) | 33333333.3 | ⏳ | ❌ 0.733x est 0.27x ref 5.1 kbit 1.9 s (target infeasible, LabelSelect, see note) | ⏸ | ⏸ | ⏸ | folding target: committed JSON folds LabelSelect (PE=5, 3 cycles); the optimizer keeps LabelSelect at PE=1 by design (fmax), 11 cycles: target infeasible |
| tfc-w1a2 (AUP-ZU3_8GB) | 12500000.0 | ❌ 0.533x 8.2 kbit 0.1 s #002 | ⏳ | ⏸ | ⏸ | ⏸ |  |
| tfc-w2a2 (AUP-ZU3_8GB) | 12500000.0 | ❌ 0.533x 6.7 kbit 0.1 s #002 | ⏳ | ⏸ | ⏸ | ⏸ |  |
| lfc-w1a1 (AUP-ZU3_8GB) | 195312.5 | ⏳ | ⏳ | ⏸ | ⏸ | ⏸ |  |
| lfc-w1a2 (AUP-ZU3_8GB) | 48828.1 | ⏳ | ⏳ | ⏸ | ⏸ | ⏸ |  |
| cnv-w1a1 (AUP-ZU3_8GB) | 12244.4 | ✅ 1.001x 11.9 kbit 1.9 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w1a2 (AUP-ZU3_8GB) | 3028.5 | ✅ 1.008x 26.7 kbit 2.9 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w2a2 (AUP-ZU3_8GB) | 3028.5 | ✅ 1.008x 25.5 kbit 3.2 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w1a2 (KV260_SOM) | 3028.5 | ✅ 1.008x 26.7 kbit 2.8 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cnv-w2a2 (U55C) | 3028.5 | ✅ 1.007x 25.5 kbit 3.2 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| gtsrb (AUP-ZU3_8GB) | 8138.0 | ✅ 1.000x 9.9 kbit 2.3 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| kws (AUP-ZU3_8GB) | 25510.2 | ✅ 0.998x 4.3 kbit 0.1 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| cybersecurity-mlp (AUP-ZU3_8GB) | 100000000.0 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| vgg10-radioml (ZCU104) | 243190.7 | ✅ 1.001x 14.2 kbit 0.4 s | ⏸ | ⏸ | ⏸ | ⏸ |  |
| espcn (KV260_SOM) | 381.5 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| mobilenet_v1 (U55C) | 422.7 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| resnet18_4w4a (U55C) | 3814.7 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| resnet18_8w8a (U55C) | 1695.4 | skip | ⏸ | ⏸ | ⏸ | ⏸ | FIFO: n/a, committed folding has runtime-writeable URAM weights, not initialized in stitched-IP rtlsim |
| quantyolov8_4w4a (U55C) | 90.4 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| quantyolov8_8w8a (U55C) | 90.4 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| bert (VCK190) | 508.6 | ❌ 0.0 kbit 26.0 s #001 | ⏸ | ⏸ | ⏸ | ⏸ |  |
| resnet50 (V80) | 276.8 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |

Issues: `claude-tools/finn-examples-features/issues/<NNN-...>/ISSUE.md` (+ `REPORT.md` of the fix agent).
* #001 bert heuristic_analytical: per-node rtlsim characterization of a node without tree model times out (dynamic MVAU suspected).
* #002 tfc: rtlsim ~15 cycles/frame vs estimate 8 (output correct); per-frame overhead at near-full unrolling suspected.
