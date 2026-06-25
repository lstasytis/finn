#!/usr/bin/env bash
# SWG (ConvolutionInputGenerator) optimizer run, seeded FROM the best candidate
# found so far so progress accumulates across restarts, with a per-worker bash
# budget so the agent iterates via eval_tree_model instead of paging through RTL.
cd /home/teckmann/finn/tools/Bespoke-Base-Retreat-26
exec .venv/bin/python examples/tav_tree_model_loop.py ConvolutionInputGenerator \
  --baseline outputs/saved_best/cig_swg_best_score19123_input14of39_20260625-1936.py \
  --oracle both --parallel 2 --max-iterations 10 --bash-budget 20
