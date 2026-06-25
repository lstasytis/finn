#!/usr/bin/env bash
cd /home/teckmann/finn/tools/Bespoke-Base-Retreat-26
exec .venv/bin/python examples/tav_tree_model_loop.py FMPadding \
  --oracle both --parallel 1 --max-iterations 3
