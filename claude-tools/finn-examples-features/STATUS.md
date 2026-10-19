# Feature-test status (models x features)

Generated 2026-10-19 12:23 by `tools/status_table.py` from the campaign results. Cells: result, rtlsim/estimate throughput ratio, (folding/DWC) own estimate vs the committed folding's estimate, FIFO storage, padding DWCs, time of the feature's step (FIFO sizing or folding; measured under load, not comparable to isolated timings). ⏳ running, ⏸ not run yet, aborted = killed at the reboot deadline.

| model (board) | est. fps (committed) | FIFO analytical | Folding target | Folding maximize | DWC | FIFO rtlsim | Notes |
|---|---|---|---|---|---|---|---|
| tfc-w1a1 (AUP-ZU3_8GB) | 33333333.3 | ⏳ | ❌ 0.733x est 0.27x ref 5.1 kbit 1.9 s #002 + target infeasible (LabelSelect) | ❌ 0.700x est 0.21x ref 3.1 kbit 20.8 s #002 | ⏸ | ⏸ | folding target: committed JSON folds LabelSelect (PE=5, 3 cycles); the optimizer keeps LabelSelect at PE=1 by design (fmax), 11 cycles: target infeasible |
| tfc-w1a2 (AUP-ZU3_8GB) | 12500000.0 | ❌ 0.533x 8.2 kbit 0.1 s #002 | ❌ 0.733x est 0.73x ref 7.9 kbit 1.9 s #002 + target infeasible (LabelSelect) | ❌ 0.696x est 0.50x ref 5.8 kbit 20.3 s #002 | ⏸ | ⏸ | LabelSelect kept at PE=1 caps the optimizer at 11 cycles (committed: 8) |
| tfc-w2a2 (AUP-ZU3_8GB) | 12500000.0 | ❌ 0.533x 6.7 kbit 0.1 s #002 | ❌ 0.733x est 0.73x ref 7.9 kbit 1.9 s #002 + target infeasible (LabelSelect) | ❌ 0.800x est 0.29x ref 2.9 kbit 21.5 s #002 | ⏸ | ⏸ |  |
| lfc-w1a1 (AUP-ZU3_8GB) | 195312.5 | ✅ 0.988x 4.8 kbit 0.6 s | ✅ 0.997x est 1.31x ref 4.5 kbit 2.2 s | ⏳ | ⏸ | ⏸ |  |
| lfc-w1a2 (AUP-ZU3_8GB) | 48828.1 | ✅ 0.997x 6.8 kbit 0.6 s | ⏳ | ⏳ | ⏸ | ⏸ |  |
| cnv-w1a1 (AUP-ZU3_8GB) | 12244.4 | ✅ 1.001x 11.9 kbit 1.9 s | ✅ 1.001x est 1.00x ref 17.5 kbit 9.9 s | ⏳ | ⏸ | ⏸ |  |
| cnv-w1a2 (AUP-ZU3_8GB) | 3028.5 | ✅ 1.008x 26.7 kbit 2.9 s | ✅ 1.014x est 1.00x ref 18.4 kbit 9.8 s | ⏳ | ⏸ | ⏸ |  |
| cnv-w2a2 (AUP-ZU3_8GB) | 3028.5 | ✅ 1.008x 25.5 kbit 3.2 s | ✅ 0.991x est 1.00x ref 13.9 kbit 9.9 s | ⏳ | ⏸ | ⏸ |  |
| cnv-w1a2 (KV260_SOM) | 3028.5 | ✅ 1.008x 26.7 kbit 2.8 s | ✅ 0.995x est 1.02x ref 21.5 kbit 11.8 s | ⏳ | ⏸ | ⏸ |  |
| cnv-w2a2 (U55C) | 3028.5 | ✅ 1.007x 25.5 kbit 3.2 s | ✅ 0.991x est 1.00x ref 17.5 kbit 10.0 s | ⏳ | ⏸ | ⏸ |  |
| gtsrb (AUP-ZU3_8GB) | 8138.0 | ✅ 1.000x 9.9 kbit 2.3 s | ✅ 1.000x est 1.09x ref 27.0 kbit 10.7 s | ⏸ | ⏸ | ⏸ |  |
| kws (AUP-ZU3_8GB) | 25510.2 | ✅ 0.998x 4.3 kbit 0.1 s | ✅ 0.998x est 1.00x ref 4.8 kbit 1.9 s | ⏸ | ⏸ | ⏸ |  |
| cybersecurity-mlp (AUP-ZU3_8GB) | 100000000.0 | ❌ 0.167x 3.7 kbit 0.1 s #002 | ⏳ | ⏸ | ⏸ | ⏸ |  |
| vgg10-radioml (ZCU104) | 243190.7 | ✅ 1.001x 14.2 kbit 0.4 s | ⏳ | ⏸ | ⏸ | ⏸ |  |
| espcn (KV260_SOM) | 381.5 | ⏳ | ⏳ | ⏸ | ⏸ | ⏸ |  |
| mobilenet_v1 (U55C) | 422.7 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| resnet18_4w4a (U55C) | 3814.7 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| resnet18_8w8a (U55C) | 1695.4 | skip | ⏸ | ⏸ | ⏸ | ⏸ | FIFO: n/a, committed folding has runtime-writeable URAM weights, not initialized in stitched-IP rtlsim |
| quantyolov8_4w4a (U55C) | 90.4 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| quantyolov8_8w8a (U55C) | 90.4 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |
| bert (VCK190) | 508.6 | ❌ 0.0 kbit 27.3 s #001 | ⏸ | ⏸ | ⏸ | ⏸ |  |
| resnet50 (V80) | 276.8 | ⏳ | ⏸ | ⏸ | ⏸ | ⏸ |  |

Issues: `claude-tools/finn-examples-features/issues/<NNN-...>/ISSUE.md` (+ `REPORT.md` of the fix agent).
* #001 bert heuristic_analytical: per-node rtlsim characterization of a node without tree model times out (dynamic MVAU suspected).
* #002 tfc, cybersecurity-mlp: correct output, but rtlsim interval a few cycles longer than the estimate (8 vs 15, 1 vs 6 cycles/frame); per-frame overhead at near-full unrolling suspected.
* #003 folding optimizer raises on branching graphs (SWG consumer = next node) for resnet18/50 and yolov8, and on bert (dynamic MVAU without weight initializer).
* #004 vgg10 padded folding: InsertDWC asserts 'DWC can only pad/crop within a row's last word' (Pool -> MVAU); not yet assigned.
