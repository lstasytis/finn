# Report 003: folding optimizer on branching graphs (resnet18/50, yolov8) and bert — fixed
(Written by the orchestrator from the fix agent's hand-back.)

## Root causes
1. Branching graphs: `_extract_swg_pair` took `graph.node[node_index + 1]` as the SWG's consumer and
   `generate_parameter_set` skipped that next node. Branches are interleaved (resnet18/yolov8 SWG ->
   SWG -> MVAU -> MVAU; resnet50 SWG -> Thresholding_rtl -> MVAU_hls; yolov8 SWG -> ElementwiseMul ->
   MVAU). Every SWG has exactly one consumer, always an MVAU (mismatched pairs: resnet18 6, resnet50 3,
   yolov8 16): the campaign's "feeds unsupported op ..." errors were all this bug.
2. bert: the attention MVAUs are `mem_mode="dynamic"` (input 1 streamed, no initializer);
   `_resize_weights` crashed on `None` at the final apply, even without padding.
3. Two InsertDWC bugs on the DWC branch (`insert_dwc.py`), visible after 1/2: (a) the datatype check
   compared the producer's output with the consumer's input 0 instead of the connected input
   (yolov8 StreamingConcat/ElementwiseAdd, bert dynamic MVAU: "Neighboring node datatypes are
   Incompatible (UINT4) != (UINT5)"); (b) a dynamic MVAU's `get_normal_input_shape(1)` is `(MW, MH)`
   without leading dims, so plain width conversions in front of the streamed operand were read as
   crops (bert padded: "... DWC can only pad/crop within a row's last word").
4. yolov8 naive TypeError: fixed by the #1725 `int()` cast.

## Fix
| commit | branch | what |
|---|---|---|
| 049a73a57 | set-folding-optimizer | SWG paired with `model.find_consumer(swg.output[0])` (real node index, `consumer_index`), paired node skipped by name; `_resize_weights` returns without initializer; no padding for MVAU/VVAU with streamed weights (`_streamed_weights`) |
| 8fe267d40 | set-folding-optimizer | tests: `test_set_folding_swg_pair_in_branching_graph`, `test_set_folding_dynamic_mvau` (both failed before) |
| 14f091ea8 | generalized-datawidthconverter | InsertDWC datatype check uses `n1.get_input_datatype(in_idx)` |
| 928e0a084 | generalized-datawidthconverter | "plain width conversion" decided on folded element counts (`np.prod(n0_out_shape) == np.prod(n1_in_shape)`) |
Landed by the orchestrator (18:45) on `feature/dev/set-folding-optimizer`,
`feature/dev/generalized-datawidthconverter` (rebased), `test/{folding,dwc}`, integration branch.
Agent branches: `fix/003`, `fix/003-dwc`; throwaway `fix/003-integ`, `fix/003-integ0`.

## Gate
Dumps under `/home/lstasytis/finn/build/finn_build/fx/fix003/dumps/`. set_folding gate vs `base1`
(frozen `fx/refactor_env_integ`): 84 runs, 82 byte-identical (incl. every run that raised before);
only `yolov8_*__naive` changed (TypeError -> ok, #1725 cast). DWC-fix gate (padded + target on the 15
small checkpoints, with/without the InsertDWC commits): 30/30 identical.
Target models with the fix (cyc = folded max_cycles, tgt = reference):
| model | target | padded | fifo | maximize |
|---|---|---|---|---|
| resnet18 4w4a | 65536 = tgt, fits | 65536 fits | 65536 fits | 65538 fits (47 min) |
| resnet18 8w8a | 147456 = tgt, budget not met | same | same | 9437184, budget not met |
| resnet50 V80 | 903168 = tgt, fits | 802823 fits | > 4 h | > 4 h |
| yolov8 4w4a | 1105920 = tgt, fits | 1105920 fits | > 4 h | > 4 h |
| yolov8 8w8a | 1105920 = tgt, fits | 1105920 fits | > 4 h | > 4 h |
| bert VCK190 | 393216 = tgt, fits | 393216 fits (needs 928e0a084) | 393216 fits | > 4 h |
Timeouts are slowness (bert maximize ~15 folds x ~520 s). resnet18 8w8a misses the BRAM budget (5027
vs committed 2254 BRAM + 404 URAM) because URAM weights need `allow_uram_weights` /
`runtime_writeable_weights`.

## Tests
`test_set_folding.py` on fix/003: 34 passed, 2 skipped, 1 xfail; on fix/003-dwc: 31 passed, 3 skipped,
1 xfail; `test_fpgadataflow_dwc.py` 88 passed. Lint clean.
Feature test `test_resnet18_folding[U55C-resnet18_4w4a-target]` (fix/003): folding ok, estimate
3814.58 fps vs target 3814, stitched IP built, **stitched-IP rtlsim verification SUCCESS** (2.5 h);
the rtlsim throughput measurement was killed at 18:38 (deadline): 0.98x check open.
Build: `fx/fix003/fold-target_resnet18_resnet18_4w4a_U55C_z68k0g76/`.

## Open points
1. Maximize / FIFO-aware mode on yolov8/resnet50/bert takes > 4 h (maximize does ~15 full folds).
2. resnet18 8w8a cannot fit U55C without URAM weights.
3. `Optimizer.dwc_cycles` compares normal shapes and over-costs DWCs in front of a dynamic MVAU's
   weight input (conservative; bert only).
4. Resume the resnet18 feature test (~5.5 h): `cd build/wt/fix003 && source .../wtenv.sh . &&
   NUM_DEFAULT_WORKERS=3 python -m pytest -p no:cacheprovider -v
   "tests/benchmark/resnet18/test_build_resnet18.py::test_resnet18_folding[U55C-resnet18_4w4a-target]"`.
5. Leftover worktrees `build/wt/fix003-integ`, `build/wt/fix003-integ0` can be removed.
