#!/usr/bin/env bash
# SWG (ConvolutionInputGenerator) optimizer run.
#  --baseline    : seed FROM the best candidate so far (progress accumulates).
#  --curriculum  : solve one case at a time (smallest first), lock it, then widen
#                  -- non-regressing, so it can build a branching tree instead of
#                  thrashing on all 39 cases at once (the 19123 local-optimum).
#  --memory      : carry the best lineage's reasoning across iterations.
#  --bash-budget : cap source-reading so it iterates via eval_tree_model.
cd /home/teckmann/finn/tools/Bespoke-Base-Retreat-26
exec .venv/bin/python examples/tav_tree_model_loop.py ConvolutionInputGenerator \
  --baseline outputs/saved_best/cig_swg_best_score19123_input14of39_20260625-1936.py \
  --oracle both --parallel 2 --max-iterations 40 --bash-budget 20 \
  --curriculum --memory
