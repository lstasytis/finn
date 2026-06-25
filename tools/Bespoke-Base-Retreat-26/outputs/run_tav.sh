#!/usr/bin/env bash
cd /home/teckmann/finn/tools/Bespoke-Base-Retreat-26
exec .venv/bin/python examples/tav_tree_model_loop.py ConvolutionInputGenerator \
  --test "tests/fpgadataflow/test_fpgadataflow_downsampler.py::test_fpgadataflow_analytical_characterization_downsampler" \
  --oracle both --parallel 2 --max-iterations 10
