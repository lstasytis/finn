#!/usr/bin/env bash
# claude-tools/e2e/run_matrix.sh — drive the full end-to-end benchmark matrix.
#
# Streams (run concurrently, one model at a time within each stream):
#   baseline  : env/baseline worktree  -> test_e2e_baseline
#   features  : env/e2e worktree       -> test_e2e_{fifo_sizing,folding,generalized_dwc,combined}
#   aligner   : main checkout (env/label-aligner) -> test_e2e_label_aligner
#
# Small models run first; mobilenet_v1 last (heaviest by far). All streams share
# one results dir + build root, so the report generator sees everything:
#   python tests/benchmark/e2e_report.py --results "$RESULTS"
#
# Usage:  claude-tools/e2e/run_matrix.sh [baseline|features|aligner|all]
set -uo pipefail

MAIN=/home/lstasytis/backup/finn
RESULTS=$MAIN/e2e_results
BUILDS=$MAIN/e2e_build
LOGS=$MAIN/e2e_logs
mkdir -p "$RESULTS" "$BUILDS" "$LOGS"

# cheapest first so results accumulate early; mobilenet last
MODELS_SMALL=(
  "bnn-pynq-tfc-w1a1" "bnn-pynq-tfc-w1a2" "bnn-pynq-tfc-w2a2"
  "cybersecurity-mlp"
  "bnn-pynq-cnv-w1a1" "bnn-pynq-cnv-w1a2" "bnn-pynq-cnv-w2a2"
  "vgg10-radioml"
)
MODELS_HEAVY=("mobilenet_v1")

run_tests() {  # run_tests <tree> <log-tag> <testfile> [testfile...]
  local tree="$1" tag="$2"; shift 2
  local files=("$@")
  for model in "${MODELS_SMALL[@]}" "${MODELS_HEAVY[@]}"; do
    for f in "${files[@]}"; do
      # test function name == module name (test_e2e_baseline.py::test_e2e_baseline)
      local tname rc
      tname=$(basename "$f" .py)
      echo "[$tag] $tname[$model]  start $(date +%H:%M:%S)"
      ( cd "$tree" &&
        FINN_ROOT="$tree" PYTHONPATH="$tree/src" \
        FINN_E2E_RESULTS="$RESULTS" FINN_E2E_BUILD_ROOT="$BUILDS" \
        NUM_DEFAULT_WORKERS=8 \
        python3 -m pytest -q -p no:cacheprovider \
          "tests/benchmark/$f::${tname}[${model}]"
      ) >> "$LOGS/matrix_${tag}.log" 2>&1
      rc=$?
      echo "[$tag] $tname[$model] done rc=$rc $(date +%H:%M:%S)"
    done
  done
}

stream_baseline() { run_tests "$MAIN/envs/baseline" baseline test_e2e_baseline.py; }
stream_features() {
  run_tests "$MAIN/envs/e2e" features \
    test_e2e_fifo_sizing.py test_e2e_folding.py test_e2e_generalized_dwc.py test_e2e_combined.py
}
stream_aligner()  { run_tests "$MAIN" aligner test_e2e_label_aligner.py; }

case "${1:-all}" in
  baseline) stream_baseline ;;
  features) stream_features ;;
  aligner)  stream_aligner ;;
  all)
    stream_baseline & B=$!
    stream_features & F=$!
    stream_aligner  & A=$!
    wait $B $F $A
    ;;
  *) echo "usage: run_matrix.sh [baseline|features|aligner|all]" >&2; exit 2 ;;
esac
echo "matrix stream(s) '${1:-all}' finished $(date)"
