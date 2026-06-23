#!/usr/bin/env bash
# AlphaEvolve-style optimization of the downsampler (ConvolutionInputGenerator)
# get_tree_model, using your local LLM + the tav_eval validator.
#
# Point it at your model with TAV_LLM_CMD (or pass --llm-cmd). The command must
# read a prompt on stdin and print a get_tree_model on stdout (see
# llm_adapter.py to instead call Bespoke directly in python).
#
# Usage:
#   TAV_LLM_CMD='python tools/Bespoke-Base-Retreat-26/generate.py' \
#       tools/tav_eval/run_evolve_downsampler.sh
#   tools/tav_eval/run_evolve_downsampler.sh --llm-cmd '...' -n 30 --apply-best
#
# Env:
#   ITERATIONS   max LLM iterations (default 20)
# Extra args are forwarded to evolve.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

cd "$REPO"
python tools/tav_eval/evolve.py ConvolutionInputGenerator \
  --test "tests/fpgadataflow/test_fpgadataflow_downsampler.py::test_fpgadataflow_analytical_characterization_downsampler" \
  -n "${ITERATIONS:-20}" \
  "$@"
