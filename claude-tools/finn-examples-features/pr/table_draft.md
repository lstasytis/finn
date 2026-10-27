| model | board | folding | rtlsim / estimate | FIFO storage | sizing time (heuristic_analytical) | sizing time (heuristic_rtlsim) | result |
|---|---|---|---:|---:|---:|---:|---|
| tfc-w1a1 | AUP-ZU3_8GB | small | 1.000 | 0.64 kB | 0.1 s | 972 s | ✅ pass |
| tfc-w1a1 | AUP-ZU3_8GB | full | 1.000 | 2.28 kB | 0.2 s | 21680 s | ✅ pass |
| tfc-w1a2 | AUP-ZU3_8GB | small | 1.020 | 0.34 kB | 4.0 s | 469 s | ✅ pass |
| tfc-w1a2 | AUP-ZU3_8GB | full | 1.000 | 1.02 kB | 0.2 s | 2679 s | ✅ pass |
| tfc-w2a2 | AUP-ZU3_8GB | small | 1.019 | 0.29 kB | 3.6 s | 467 s | ✅ pass |
| tfc-w2a2 | AUP-ZU3_8GB | full | 1.000 | 0.84 kB | 0.2 s | 2249 s | ✅ pass |
| cnv-w1a1 | AUP-ZU3_8GB | small | 1.001 | 1.73 kB | 2.5 s | 1583 s | ✅ pass |
| cnv-w1a1 | AUP-ZU3_8GB | full | 1.001 | 1.49 kB | 3.0 s | 4581 s | ✅ pass |
| cnv-w1a2 | AUP-ZU3_8GB | small | | | | 5331 s | ⏳ PENDING (rerun at the final commit) |
| cnv-w1a2 | AUP-ZU3_8GB | full | 1.008 | 3.33 kB | 6.5 s | 5604 s | ✅ pass |
| cnv-w1a2 | KV260_SOM | small | | | | 5486 s | ⏳ PENDING (rerun at the final commit) |
| cnv-w1a2 | KV260_SOM | full | 1.008 | 3.33 kB | 5.7 s | 5846 s | ✅ pass |
| cnv-w2a2 | AUP-ZU3_8GB | small | | | | 5408 s | ⏳ PENDING (rerun at the final commit) |
| cnv-w2a2 | AUP-ZU3_8GB | full | | | | | ⏳ PENDING |
| cnv-w2a2 | U55C | small | | | | 5466 s | ⏳ PENDING (rerun at the final commit) |
| cnv-w2a2 | U55C | full | | | | | ⏳ PENDING |
| lfc-w1a1 | AUP-ZU3_8GB | small | 1.000 | 0.44 kB | 4.5 s | 10036 s | ✅ pass |
| lfc-w1a1 | AUP-ZU3_8GB | full | 1.002 | 0.60 kB | 1.0 s | 7999 s | ✅ pass |
| lfc-w1a2 | AUP-ZU3_8GB | small | 1.000 | 0.69 kB | 4.5 s | 23815 s | ✅ pass |
| lfc-w1a2 | AUP-ZU3_8GB | full | 1.000 | 0.85 kB | 1.0 s | 11986 s | ✅ pass |
| gtsrb | AUP-ZU3_8GB | small | | | | 1063 s | ⏳ PENDING (rerun at the final commit) |
| gtsrb | AUP-ZU3_8GB | full | 1.000 | 1.23 kB | 3.4 s | 3972 s | ✅ pass |
| kws | AUP-ZU3_8GB | small | 1.000 | 0.53 kB | 0.2 s | 419 s | ✅ pass |
| kws | AUP-ZU3_8GB | full | 1.000 | 0.53 kB | 0.3 s | 730 s | ✅ pass |
| cybersecurity-mlp | AUP-ZU3_8GB | small | 1.100 | 0.24 kB | 0.1 s | 2163 s | ✅ pass |
| cybersecurity-mlp | AUP-ZU3_8GB | full | 1.333 | 0.46 kB | 0.2 s | 11023 s | ✅ pass |
| vgg10-radioml | ZCU104 | small | 1.001 | 1.28 kB | 24.2 s | 1164 s | ✅ pass |
| vgg10-radioml | ZCU104 | full | 1.001 | 1.93 kB | 1.9 s | 1218 s | ✅ pass |
| mobilenet_v1 | U55C | small | 0.991 | 64.88 kB | 355 s | 23790 s | ✅ pass |
| mobilenet_v1 | U55C | full | 1.000 | 17.01 kB | 144 s | 8197 s | ✅ pass |
| resnet18_4w4a | U55C | small | | | | 7321 s | ⏳ PENDING (rerun at the final commit) |
| resnet18_4w4a | U55C | full | | | | 3496 s | ⏳ PENDING (rerun at the final commit) |
| resnet50 | V80 | small | | | |  | ⏳ PENDING (rerun at the final commit) |
| resnet50 | V80 | full | | | |  | ⏳ PENDING; last run: 1.000× throughput, wrong output from the folding config (not FIFO, [below](#failures)) |
| bert | VCK190 | small | 1.000 | 913.76 kB | 1664 s | 6656 s | ✅ pass |
| bert | VCK190 | full | 1.000 | 909.89 kB | 49.7 s | 2027 s | ✅ pass |
| espcn | KV260_SOM | small | | | | | ⏳ PENDING |
| espcn | KV260_SOM | full | | | | | ⏳ PENDING |
| quantyolov8_4w4a | U55C | small | | | | | ⏳ PENDING (rerun with the Concat join window floor) |
| quantyolov8_4w4a | U55C | full | | | | | ⏳ PENDING |
| quantyolov8_8w8a | U55C | small | | | | | ⏳ PENDING |
| quantyolov8_8w8a | U55C | full | | | | | ⏳ PENDING |
| transformer_benchmark | ZCU104 | small | | | | 1654 s | ⏳ PENDING (rerun at the final commit) |
| transformer_benchmark | ZCU104 | full | | | | 2022 s | ⏳ PENDING (rerun at the final commit) |
| transformer_radioml | ZCU104 | small | | | | | ⏳ PENDING (no-deadlock check only: rtlsim must finish at ≥ 0.98× the estimate) |
| transformer_radioml | ZCU104 | full | | | | | ⏳ PENDING (no-deadlock check only: rtlsim must finish at ≥ 0.98× the estimate) |
| transformer_vision | ZCU104 | small | | | | | ⏳ PENDING (no-deadlock check only: rtlsim must finish at ≥ 0.98× the estimate) |
| transformer_vision | ZCU104 | full | | | | | ⏳ PENDING (no-deadlock check only: rtlsim must finish at ≥ 0.98× the estimate) |
| transformer_language | VCK190 | small | | | | | ⏳ PENDING (no-deadlock check only: rtlsim must finish at ≥ 0.98× the estimate) |
| resnet18_8w8a | U55C | small, full | | | | | not run: runtime-writeable URAM weights are not simulated |
