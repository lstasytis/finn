# Issue 015 report: FIFO skip-edge deadlock (radioml, vision, language, bert) — fixed
(Saved by the orchestrator from the agent's hand-back; subagents cannot write report files.)

## Summary
* Issue 013's latency change alone is not enough: radioml committed/small shortcut 172/122 (bounds 47/95) avoids
  the hang but radioml small ran at 0.63x (only that FIFO at 190 -> 1.0x); vision small shortcut 8 (bound 191) ->
  deadlock; language small shortcut 2 -> hang; bert small fold-target attention shortcut 2 (its fork is fed directly
  by the graph input, which the timeline pass skips).
* Fix (fix/015 on feature/dev/analytical-fifo-sizing af38b555c): `join_window_floors` in `DeriveFIFOSizes`. For every
  two-stream join it walks back to the shared fork and takes the larger of a deadlock bound (fork words the other
  branch must read before join token k) and a paced bound (fork writes at the graph period, each node emits word j
  once its window is in and no faster than its get_exp_cycles()), plus 25 %, over two frames. New HWCustomOp hooks:
  `get_tav_input_window()` (input words a block of output needs; default frame ratio in lowest terms; MVAU in0 one
  vector or TH vectors when tiled; HWSoftmax a row; InnerShuffle a page; dynamic MVAU in1 the whole matrix) and
  `get_tav_input_need()` (explicit per-word need of reordering nodes; OuterShuffle from its permutation).
* radioml's earlier 0.2x was an estimate problem removed by the Split/Concat dataflow + issue 010 changes (fold-target
  full on the current integration: 16271 vs 16276 fps). heuristic_rtlsim's radioml failure = the Split/Concat node
  harness, fixed meanwhile by issue 016 (af38b555c); radioml small heuristic_rtlsim now passes 1.0x.

## Commits on fix/015 (base af38b555c)
729e7d033 join floor (deadlock bound) + get_tav_input_window hook (HWCustomOp, MVAU, HWSoftmax, InnerShuffle) + test;
b0fbc933e paced bound + test; 57aaa6e98 docs "Join window floor"; 8aeb70760 get_tav_input_need hook, OuterShuffle need
from its permutation (bert) + test; 99de01f40 docs; 4655649a7 tiled MVAU window = TH vectors (bert FFN); 39dac9a18 docs.
test_fifosizing_chained_tav.py 7 passed; new test_join_window_floor (residual MVAU SIMD=PE=1: TAVs 2 -> floor 80; two
MVAU_rtl in series: floor 79 < TAVs 90; OuterShuffle need table). Lint clean; merges into the integration branch
without conflicts, DWC invariant holds.

## Evidence (stitched-IP rtlsim, scripts/perf_override.py)
| build | join input | sized | floor (backlog) | result |
|---|---|---|---|---|
| radioml committed, old sizer | MLP shortcut | 20 | 113 (deadlock 47) | hang; at 128 completes |
| radioml fold-target (fix011 va3bwjc7) | MLP shortcut | 37 | 122 (47) | completes |
| radioml small, 013 sizer | MLP shortcut | 122 | 239 (deadlock 95, paced 191) | 0.63x; only this FIFO at 190 -> 1.00x |
| vision fold-target | MLP shortcuts | 92 / 108 | 419-475 (deadlock 191) | hang; at 512 completes |
| bert small fold-target | attention shortcut | 2 | 118545 (OuterShuffle need) | hang; only this FIFO at 98304 -> 1.00x |
| bert small committed | FFN shortcut | 4566 | 5813 (TH window) | 0.82x; at 5813 or 9132 -> 1.00x |

## Feature tests on fix/015-int3 (= dev-finnexamples-features 18f55a51c + fix/015; logs fx/fix015-int3/logs/)
| test | before | after | FIFO kbit |
|---|---|---|---|
| radioml fifo small / full | 0.63x / - | PASSED 1.00 / PASSED 1.00 | 313.5 / 193.1 |
| radioml fold-target small / full | - / 0.2x | PASSED 1.00 / PASSED 1.00 | 329.7 / 488.4 |
| radioml heuristic_rtlsim small | node rtlsim fails | PASSED 1.00 | 346.0 |
| vision fifo small / full | hang | PASSED 1.00 / PASSED 1.00 | 719.2 / 1242.4 |
| vision fold-target small / full | hang | PASSED 1.00 / PASSED 1.00 | 741.6 / 1310.8 |
| bert fold-target small | hang | PASSED 1.00 | 6425.3 |
| bert fifo small | hang | PASSED 1.00 (0.82x before 4655649a7) | 7311.3 |
| language fifo small | hang | no hang, output WRONG (12.6 % > atol 0.5, max 4.56) | - |

## Total FIFO size (offline, scripts/compare_all.sh; base = feature-branch sizer)
bnn-pynq (9), cybersecurity-mlp, gtsrb, kws, mobilenet_v1, vgg10, espcn, resnet18, yolov8 (2): 1.000; resnet50 1.002;
transformer_benchmark 1.047 (attention shortcut 2140 -> 2698, needs 2065-2077); radioml / vision / language 1.025 /
1.026 / 1.026; bert 1.182 (attention shortcut 21338 -> 30303, 128 bit; 21338 passes). The floor over-estimates
whole-frame attention shortcuts (paced at node rates + 25 %); within the 1.5x guidance, a smaller margin on large
backlogs would win most of it back.

## Open points
language small: output wrong once the hang is gone (not FIFO related). The floor covers two-stream joins only;
multi-input characterized nodes (dynamic MVAU, Concat) rely on the timeline pass; ops without a window hook
(LayerNorm, VVAU, SWG) are treated as one-to-one. The timeline pass still skips a fork fed by the graph input.

## Branches / deletions
fix/015 (build/wt/fix015; fix/015-v1 backup), fix/015-int3 (verification), throwaway fix/015-int, fix/015-int2.
Deleted own code_gen/tav/vivado/rtlsim dirs in fx/fix015-int{,2,3} (~15 GB); fix015-int3 still 14 GB.
