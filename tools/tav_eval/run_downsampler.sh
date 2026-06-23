#!/usr/bin/env bash
# Run the tav_eval harness on the downsampler characterization test.
#
# The downsampler is implemented as a ConvolutionInputGenerator node, so this
# splices the ConvolutionInputGenerator get_tree_model candidate and runs the
# downsampler analytical-characterization test. With the committed cached_models
# rtlsim references present, rtlsim is skipped (cache hit) and the per-case TAV
# delta log is printed.
#
# Usage:
#   tools/tav_eval/run_downsampler.sh                  # baseline candidate (identity)
#   tools/tav_eval/run_downsampler.sh my_candidate.py  # evaluate your candidate
#
# Any extra args after the candidate are forwarded to tav_eval.py.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CANDIDATE="${1:-$HERE/examples/convolutioninputgenerator_tree_model.py}"
shift || true

cd "$REPO"
python tools/tav_eval/tav_eval.py ConvolutionInputGenerator "$CANDIDATE" \
  --test "tests/fpgadataflow/test_fpgadataflow_downsampler.py::test_fpgadataflow_analytical_characterization_downsampler" \
  -v "$@"
