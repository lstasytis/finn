# Issue 016 report: Split/Concat tree models vs the dataflow pragma — fixed
(Saved by the orchestrator from the agent's hand-back; subagents cannot write report files.)

## Measured (node rtlsim, all streams traced, 5 frames; scripts/measure.py, analyze.py)
7 configs per op (ChannelsPerStream 8,8,8,8 at SIMD 1/2/4; 4,8,12 SIMD 2 [1,3]; 16,4 SIMD 1 [1,2,2]; 6,2,10 SIMD 2
[1,4]; 2,6 SIMD 2 [1,5]), each with and without the pragma.
| | no pragma (plain dev) | `#pragma HLS dataflow` |
|---|---|---|
| interval | 5 | 1 |
| Split: input word k / out0 word j | 1+5k / 7+5(off+j) | 1+k / 4+(off+j) (5 instead of 4 in 4 of 7 configs) |
| Concat: in0 word j / output word k | 6+5(off+j) / 2+5k | 3+(off+j) / 4+k (2 words prefetched per input at start) |
| rtlsim cycles, 5 frames of W words | 25W+4 | 5W+5 (Split 5W+6 in some configs) |
| old tree model vs rtlsim | 0/0 in all 7 configs, both ops | wrong: period 5x, deltas up to 129 tokens |
| new tree model vs rtlsim (up to rotation) | 0/0 | Concat 0/0; Split 0/1-2 |

## Fix: af38b555c on fix/016-fifo (fast-forward of feature/dev/analytical-fifo-sizing 86942e9ee)
* `StreamingSplit/Concat.get_tree_model_timing()` -> (interval, first read, first write); base classes keep the
  unpipelined values (5,1,7) / (5,6,2) (same arrays as before); `StreamingSplit_hls` / `StreamingConcat_hls` return
  (1,1,4) / (1,3,4) if `self.has_dataflow_top()`.
* New `HLSBackend.has_dataflow_top()` (12 lines): runs the op's own `pragmas()`, checks for `#pragma HLS dataflow`,
  restores `code_gen_dict`. The model follows the generated code, so each PR is correct alone and both merged in
  either order; the pragma PR stays a pure HLS change; no integration-only commit.
* Node-rtlsim TAV derivation (`heuristic_rtlsim`) never worked for these ops (default io_dict drives in0 / drains out0
  only: Split crashed on the multi-output count dict, Concat timed out); new `derive_token_access_vectors` overrides
  drive all inputs and drain all outputs (stored rows still in0/out0).
* New `tests/fpgadataflow/test_fpgadataflow_split_concat_characterization.py` (separate file to avoid conflicts with
  the pragma PR's test hunks): 7 configs each, `tree_model_test_up_to_rotation`, tolerance 0.5 % + 2.

## Tests
fix/016-fifo (no pragma): 14 passed, exact. fix/016-integ (integration + fix, pragma): 14 passed; negative control
(dataflow case disabled = old model): 14 failed (length delta -254 / -286). Functional split/concat + pragma exp-cycles
+ chained_tav: 32 passed. Lint clean. transformer_benchmark small on fix/016-integ: fifo_sizing[heuristic_analytical]
PASS 1.000 (91.7 kbit), [heuristic_rtlsim] PASS 1.000 (100.3 kbit, Split/Concat now characterized), folding[target]
PASS 1.000. A/B vs unfixed integration: one edge differs (2140 -> 2137), the Split/Concat do not set the throughput
there; the model matters where a Split/Concat sets the period or sits on a reconvergent path.

## Open points (not changed)
* Tree models emit one row (out0/in0) applied to every stream: for unequal ChannelsPerStream, streams i>0 get stream
  0's token count/timing (Concat stream i is read ii*offset_i later) -> can under-size if C_i > C_0. Predates the
  pragma; all finn-examples Split/Concat are equal-width. Fix: one row per stream.
* `StreamingSplit.get_number_output_values` uses `folded_output_shape[1:-1]` (drops the first dim) -> wrong count if
  numInputVectors has no leading 1 (upstream code, real models always have batch 1).
* With the pragma the first out0 word trails its input by 3-4 cycles (model: 3), within TAV_PIPELINE_LATENCY.

## Housekeeping
Kept worktrees fix016-fifo, fix016-integ (verification), build dir fx/fix016-integ (3.0 GB). Deleted scratch worktree
fix016-base, fx/fix016-base (0.7 GB), fx/fix016-fifo (1.0 GB). Scripts: scripts/{measure,analyze,retree,oldio,fifocmp}.py;
raw traces in the session scratchpad i016/.
