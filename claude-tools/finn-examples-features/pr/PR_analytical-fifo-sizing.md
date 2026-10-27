# Analytical FIFO sizing: a single chained token-access-vector sizer, verified deadlock-free on finn-examples

**Merge:** `feature/dev/analytical-fifo-sizing` → `feature/analytical-fifo-sizing` (the head branch of Xilinx/finn#1185).
This is a fast-forward: the new branch contains every commit of the current PR branch.

## Summary

This brings #1185 up to date with `upstream/dev` (as of 2026-10-19) and makes the analytical FIFO sizer
(`auto_fifo_strategy="heuristic_analytical"`) safe to use on real models. It was tested on
<!--N_MODELS-->24<!--/N_MODELS--> finn-examples networks, each with its committed folding (`full`) and with a smaller
folding (`small`, parallelism divided by 4 or 8):

* **No deadlocks and no throughput loss.** <!--N_PASS-->XX<!--/N_PASS--> of <!--N_RUN-->XX<!--/N_RUN--> model/folding
  pairs finish stitched-IP rtlsim at ≥ 0.98× the build's own analytic estimate. All except the three
  transformer models that are broken upstream also match the golden output. The one wrong output (resnet50
  full) comes from a folding-config bug, not from the FIFOs ([Failures](#failures)).
* **Sizing takes seconds to minutes.** Each node's token access vector (TAV, its cycle-by-cycle read/write
  schedule) comes from a Python tree model of the node, so no simulation is needed except for the few node
  types without a model. The same sizer fed with node-rtlsim TAVs (`heuristic_rtlsim`) takes minutes to hours.
* **One sizing method instead of five.** The pairwise strategies and `chain_composed` are gone; the
  remaining chained-TAV pass is split into named steps. Its rules are documented with measured examples in
  `docs/finn/fifo_sizing_heuristics.rst`.

**What to review:** `src/finn/transformation/fpgadataflow/derive_characteristic.py` (the sizer), the
`get_tav_*` hooks and tree models in the custom ops, `build_dataflow_config.py` (removed options), the new
tests in `tests/fpgadataflow/` and the docs page. The rest of the file list is the `upstream/dev` merge and
the per-model benchmark tests ([Also in this diff](#also-in-this-diff)).

## Results on the finn-examples models

Each model's FIFOs are sized with `heuristic_analytical`; the stitched IP is then simulated (XSI rtlsim) on
the model's golden input. A row passes if the output equals the golden output and rtlsim reaches ≥ 0.98× the
estimate, the per-model test's own assertion.

* **folding:** `full` is the committed finn-examples folding. `small` divides every node's PE/SIMD by 4
  (bnn-pynq, kws, cybersecurity-mlp, gtsrb) or 8 (all others), using only values that SetFolding's optimizer
  allows for that op. This moves the bottleneck and makes the branches of residual graphs unbalanced.
* **rtlsim / estimate:** measured frames per second over FINN's analytic estimate for the same build. 1.000 means
  the FIFOs cost no throughput; values above 1 mean the estimate is pessimistic for that model.
* **FIFO storage:** sum over all inserted FIFOs of depth × stream width (1 kB = 1024 bytes).
* **sizing time:** wall-clock of `step_set_fifo_depths`, with tree-model TAVs (`heuristic_analytical`) and with
  node-rtlsim TAVs (`heuristic_rtlsim`). The analytical time includes node rtlsim for node types without a tree
  model, which is why bert small and resnet50 small take ~25 minutes. Times come from a shared 128-thread
  machine under load: read them as orders of magnitude.

<!--TABLE-->
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
<!--/TABLE-->

The table was produced by the finn-examples integration branch `dev-finnexamples-features`: this branch plus
the remaining finn-examples models (bert, espcn, resnet18, transformer, yolov8) and the `small` foldings. That
branch is pushed separately and is not part of this PR. This PR carries the per-model tests for bnn-pynq,
cybersecurity-mlp, gtsrb, kws, mobilenet_v1, resnet50 and vgg10-radioml (committed foldings only).

### Failures

<!--FAILURES-->
**resnet50, full folding (V80): wrong output, not caused by the FIFOs.** The committed folding sets the global
average pool `Pool_hls_1` to PE 4 but has no entry for its sliding window generator
`ConvolutionInputGenerator_rtl_21`, which therefore keeps SIMD 2048. The data width converter inserted between the
two does not reorder, so the pool adds up words from the wrong channels. A node-level rtlsim of just these three
nodes reproduces the wrong sums exactly, and gives the correct result with SWG SIMD = 4. Throughput is
unaffected: rtlsim runs at 1.000× the estimate. Follow-up outside this PR: add
`"ConvolutionInputGenerator_rtl_21": {"SIMD": 4}` to finn-examples' `resnet50_folding_config_V80.json`.

**transformer_radioml, transformer_vision, transformer_language: output not checked.** These models are known to
be broken upstream, so only deadlock freedom is claimed for them: rtlsim must finish at ≥ 0.98× the estimate.
Their golden-output check is ignored.

**Not run:** resnet18_8w8a, whose runtime-writeable URAM weights the stitched-IP rtlsim does not load.
<!--/FAILURES-->

## What changed compared with the current PR

### User-facing

* `auto_fifo_strategy` keeps `heuristic_analytical` (TAVs from the nodes' Python tree models, with node
  rtlsim only for node types that have no model), `heuristic_rtlsim` (TAVs from node rtlsim for every node)
  and `largefifo_rtlsim`. As in the current #1185, dev's `characterize` strategy is replaced by
  `heuristic_rtlsim`; configs that set `characterize` must switch.
* **Removed:** `heuristic_fifo_sizing_method` / `HeuristicFifoSizingMethod` and its values
  `conservative_relaxation`, `aggressive_relaxation`, `no_relaxation` and `chain_composed`. `chained_tav` was
  already the default and is now the only method. Its knobs are gone. The remaining constants are module
  constants in `derive_characteristic.py`: `PACER_TOLERANCE` 0.995, `THROTTLED_CAP` 256, `CAP_MARGIN` 1.25,
  `TAV_PIPELINE_LATENCY` 32, `JOIN_FLOOR_MARGIN` 0.25.
* `derive_characteristic.py` goes from 2333 to 1115 lines. Three rules that changed no depth on any
  regression model were dropped. The pass is a sequence of named steps (`read_node_tavs`, `node_periods`,
  `pacer_drivers`, `reconvergent_branch_nodes`, per node `local_clock` / `token_times` / `node_latency`, per
  edge `peak_occupancy` / `edge_depth`, then `join_window_floors`) instead of one function.

### Sizer rules added

Each rule was found on a finn-examples model that deadlocked or lost throughput. Terms used below:

* **pacer:** the slowest node, which sets the graph's period (cycles per frame).
* **throttled** producer: one that is held back by a slower node upstream, so it cannot run at its own rate.
* **burst floor:** the depth that absorbs a consumer's largest read burst above its average rate.
* **join:** a node with two or more input streams (AddStreams, ElementwiseAdd, StreamingConcat); its inputs
  come from a shared **fork** (DuplicateStreams / StreamingSplit).
* **`CAP_MARGIN`** (1.25): the safety factor on the period when deciding whether slow stages can take turns.

| rule | problem it fixes | example (edge depth before → after, effect) |
|---|---|---|
| **Pacer-bound chains** and **refill floor** | Balanced foldings lost 2–5 %: the sizer gave up backlog that a node feeding the pacer needs, e.g. a sliding window generator refilling rows at a frame boundary. | cnv-w1a2 KV260: `DWC_3 → MVAU_hls_1` 4 → 184; interval 13262 → 12551 cycles (= estimate) |
| **Node latency** | A TAV records a node's steady state, so no pipeline latency reached the sizer, and the slower path into a join came out hundreds of cycles short at fast foldings. Characterized nodes' writes are now delayed by the constant `TAV_PIPELINE_LATENCY` = 32 cycles (rtlsim measured 1–69) plus a per-node extra (`get_tav_extra_latency`). Each edge takes the maximum over the runs with and without this delay, so no edge shrinks. | transformer_benchmark (maximum folding): shortcut 1232 → ≥ 1501 |
| **Phase lead** | An `MVAU_rtl` TAV can open with the previous feature map's writes; the sizer took them as outputs ahead of their inputs, so residual shortcuts came out short. | transformer_benchmark: shortcut 1985 → 2077; 0.965× → 1.00× |
| **Join window floor** | A fork stalls once one branch's FIFO is full. If the other branch must read many fork words before its first output reaches the join (a wide DWC, an MVAU input vector, a softmax row, a shuffle page), a short shortcut FIFO deadlocks. `join_window_floors` walks back from each join to the shared fork and floors each join input at the larger of the deadlock backlog and the backlog when the fork writes at the graph period, plus 25 %. New hooks: `get_tav_input_window()`, `get_tav_input_need()` and a tiled-MVAU group window. | transformer_radioml: MLP shortcut 20 → 239 (hang → 1.00×); bert: attention shortcut 6785 → 118545 (hang → 1.00×), FFN shortcut 4566 → 5813 (0.82× → 1.00×) |
| ↳ **extended to StreamingConcat and sliding windows** | yolov8's PAN joins branches through `StreamingConcat`, whose inputs carry frames of different sizes, and its branches contain FMPadding, sliding window generators and nearest-neighbour upsampling, which read many input words before their first output. The floor now covers Concat (one join token = the gcd of the input frame sizes), and these nodes report the input each output word needs through `get_tav_input_need`. | quantyolov8_4w4a small: shortcut into `StreamingConcat_hls_12` 9086 → 13842 (no output in 383 M cycles → finishes). resnet18, resnet50 and yolov8 FIFOs grow by 0.2–6 %; other models are unchanged. |
| **Decoupled chain** | The burst floor kept a whole row of backlog at every fast edge of long chains, even where the slow stages around the edge already take turns behind a shallow FIFO. Where the slowest node above the producer and the slowest node at or below the consumer together fit in `CAP_MARGIN` periods, the burst floor is dropped. | mobilenet_v1 U55C: 97.2 → 17.0 kB at the same interval (788544 cycles) |
| **Throttled cap only where slow stages take turns** | Edges behind a throttled producer were capped at `THROTTLED_CAP` = 256 words. At small foldings a sliding window generator needs more backlog than that to refill at a frame boundary. The cap now applies only under the decoupled-chain condition above. | cnv-w2a2 small: 0.980× → 1.000×; cnv-w1a2 small: 0.982× → 1.000× |

### Characterization (TAV sources)

* **Dynamic MVAU** (attention matmuls, `mem_mode="dynamic"`): there was no tree model, and the rtlsim fallback
  drove only `in0`, so node characterization timed out (bert). These MVAUs now have a tree model with an `in1` row
  (one matrix burst per matrix), and the rtlsim fallback also drives `in1`.
* **Tiled MVAU** (`TH > 1`): the tree model uses the group-stall model shared with `get_exp_cycles`.
  Foldings whose output reorder buffer stalls are left to rtlsim.
* **StreamingSplit / StreamingConcat**: the tree models follow the generated HLS top: 1 word/cycle with
  the dataflow pragma, interval 5 without it. Node rtlsim now drives every input and drains every output.

### Estimates the sizer relies on

These two fixes also apply to `dev` on their own; separate PRs against `dev` are to be opened.

* `StreamingDataWidthConverter.get_exp_cycles` counts words instead of elements. Before, every DWC looked
  like the bottleneck: cnv-w1a1 was estimated at 903168 cycles per frame for an 8167-cycle accelerator.
* Tiled `MVAU_rtl.get_exp_cycles` drops the extra factor of `TH` (taken from upstream `fix/exp_cycles`)
  and models the per-group stalls of the tiled MVU. bert's estimate was 4× pessimistic.

### Tests and docs

* `tests/fpgadataflow/test_fifosizing_chained_tav.py` (new): small graphs whose TAVs all come from tree
  models (an MVAU chain behind a bursty pacer, a fork/MVAU/join block, residual graphs with `MVAU_rtl`, the
  decoupled-chain cases), sized without synthesis or rtlsim, with the expected depths pinned. Also unit checks
  of the sizer steps.
* `tests/fpgadataflow/test_fpgadataflow_split_concat_characterization.py` (new): tree model vs node rtlsim
  TAVs for StreamingSplit/Concat. `test_fpgadataflow_mvau.py`: dynamic and tiled MVAU characterization, and
  tiled `get_exp_cycles` against measured rtlsim cycles.
* Per-model tests `tests/benchmark/<model>/test_build_<model>.py` (new, 45 files, commit 19a031d71):
  `test_<model>_fifo_sizing[heuristic_analytical|heuristic_rtlsim]` resumes a cached estimate-only build of
  the committed folding, sizes the FIFOs, builds the stitched IP, and checks the golden output and ≥ 0.98× the
  estimate in rtlsim.
* `docs/finn/fifo_sizing_heuristics.rst` (new, linked from the end-to-end flow and hardware build pages): what
  a TAV is, the chained pass step by step, one section per rule with the depth it changes on a finn-examples
  model, and the throughput cost of removing each rule. It also compares the sizer's storage per FIFO with
  the `largefifo_rtlsim` reference.

### Also in this diff

* The merge of `upstream/dev` (2026-09-23 → 2026-10-19): builder and docs changes, new custom ops and their
  tests. These commits are already in `dev`.
* `tests/benchmark/`: the per-model tests above (7357 lines). They are not in `dev` and belong to this PR's
  verification.

## How to reproduce

```bash
# unit tests: no Vivado, a few minutes
pytest tests/fpgadataflow/test_fifosizing_chained_tav.py tests/fpgadataflow/test_fifosizing.py
# one model end to end: needs Vivado/Vitis 2024.2 and the models
# (tests/benchmark/models/download_models.sh); under an hour for gtsrb
pytest -m fifo_sizing -k heuristic_analytical tests/benchmark/gtsrb/test_build_gtsrb.py
```

## Known limits

<!--LIMITS-->
* Node types without a tree model are still characterized by node rtlsim (e.g. some tiled MVAU foldings).
  This is why sizing took ~25 minutes on bert small and resnet50 small.
* FIFOs are sized for throughput, not for minimum storage. On the small models the sizer can use 2–5× the
  storage of the `largefifo_rtlsim` reference, mostly in the graph-input FIFO and burst floors. The docs'
  comparison table lists every difference.
<!--/LIMITS-->

🤖 Generated with [Claude Code](https://claude.com/claude-code)
