# Report 008: folding optimizer slow on transformers / branching graphs — fixed
(Orchestrator, 2026-10-20. User: transformers must fold as fast as regular models; a branching graph
must be split into small partitions like a chain, one branch at a time, all at the same throughput.)

## Root causes (profile: `tools/fold_profile.py`, one target-mode fold, effort 50)
1. `Optimizer.fifo_resources` (part of every annealer evaluation) walked the whole graph upstream of
   the partition and recomputed every node's `get_exp_cycles` with qonnx's linear `find_producer`:
   O(N)..O(N^2) per evaluation. bert's `OuterShuffle.get_exp_cycles` simulates (~0.1 s/call), so one
   evaluation took 430 ms (99 % in `upstream_cycles`) vs 9 ms on MobileNet.
2. `_refine` (repair + polish) re-scored the whole graph for every single-step move: InsertDWC +
   SpecializeLayers (two model copies) + every node's estimates, ~0.66 s per move on resnet50,
   O(#params) moves -> quadratic (yolov8: 663 s of 1013 s).
3. Partitions were runs of consecutive graph-node indices: in a branching graph a partition mixed
   nodes of different branches.

## Fix (feature/dev/set-folding-optimizer `4b09bdfd3`, + `81b625ae3`)
* `Optimizer.exp_cycles`: `get_exp_cycles` memoized by node attributes; `fifo_resources(cache=)`:
  producer map + upstream cycles of the nodes the partition cannot reach, kept per partition.
* `_partitions` / `_chain_order`: partitions follow producer -> consumer chains (a chain continues
  into the first consumer whose producers are all placed, so the order stays topological; nodes
  without parameters are looked through); a branch never shares a partition. A linear graph gives
  exactly the old partitions.
* `_local_scorer` / `_edge_dwcs`: repair/polish moves are scored per node and per edge (InsertDWC +
  SpecializeLayers on a 2-node model of the edge, memoized by both nodes' attributes); the final
  folding is scored once on the whole graph (verdict numbers unchanged). FIFO-aware mode and graphs
  with unspecialized nodes (other than Shuffle) keep whole-graph scoring.
* `81b625ae3`: the optimizer also folds generic `Shuffle` (as dev's naive folder since upstream
  94497dee2); upstream's `test_shuffle_config_consolidation[set_folding]` fails without it.

## Evidence
* Exactness (`tools/local_score_check.py`): local == whole-graph scoring (cycles, bottleneck names,
  resources within 1e-9) on every refine evaluation: cnv, tfc, gtsrb, kws, mobilenet, resnet18,
  resnet50, yolov8, bert (unpadded, test/folding) and vgg10, mobilenet, tfc, resnet18, bert padded
  (test/dwc): ~3700 evaluations, 0 mismatches.
* Dump gate (`agents/folding_refactor/dump_folding.py`, dumps in `fx/fix008/dumps/{base,new}`):
  82 of 87 runs byte-identical (every linear model, all modes); the differences are the branching
  graphs (new partitions): resnet18 4w4a target 65536 cyc both, LUT +0.8 %, BRAM -22, DSP -40;
  resnet18 8w8a target (does not fit either way) DSP +12, LUT +1.8 %; padded runs on the
  folding-only tree crash in both versions (padding DWCs need the DWC branch).
* Time per fold (target mode), before -> after: cnv 9 -> 6 s, mobilenet 59 -> 24 s, resnet18 136 ->
  31 s, bert 509 -> 14 s, resnet50 429 -> 106 s, yolov8 1013 -> 139 s (0.3-0.5 s per node on all).
* `test_set_folding.py` 39 passed (folding branch), unit tests on test/folding and the integration
  branch.
