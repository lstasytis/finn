#!/bin/bash
# Rebuild the integration branch and/or the local test/* branches in their worktrees.
#   rebuild_branches.sh integ|fifo|folding|dwc ...
# integ = dev-finnexamples-features = DWC branch + FIFO branch + upstream #1725 + feature tests
# test/fifo    = FIFO branch + #1725 + tests
# test/folding = folding branch + #1725 + FIFO branch + tests
# test/dwc     = DWC branch (stacked on folding) + #1725 + FIFO branch + tests
# Tests (tests/benchmark, setup.cfg) are copied from $TESTS_COMMIT, default: the integration
# branch's tip before the rebuild. SUFFIX=2 builds test2/<f> in build/wt/<f>2 (created if
# missing), leaving test/<f> and its worktree to running jobs.
set -e
M=/home/lstasytis/finn
S=${SUFFIX:-}
FIFO=feature/dev/analytical-fifo-sizing
FOLD=feature/dev/set-folding-optimizer
DWC=feature/dev/generalized-datawidthconverter
PR=upstream/update/finn-examples
HLS=feature/dev/hls-exp-cycles-overhead   # standalone fix of dev's HLS cycle estimates (issue 002)
MVC=feature/dev/mvau-exp-cycles   # tiled MVAU exp cycles: upstream c15fb8a9a + group stalls (issue 009)
# separate dev PRs (issue 012): MVU output-queue overflow fix, Split/Concat dataflow pragma
DEVPRS="feature/dev/mvu-output-queue feature/dev/split-concat-dataflow"
# issue 011's upstream fixes (GAP non-square, Requant_rtl signed only) are cherry-picked from fix/011:
GAPREQ="d98fb4183 ffa6c0a5a"
# known-good merge of DWC + FIFO for the file git merges silently wrong (HANDOFF section 2)
GOOD_DWC_FILE=${GOOD_DWC_FILE:-origin/dev-finnexamples-features}

merge() {  # merge $1, finishing rerere-resolved conflicts
  if ! git merge -q --no-edit "$1" >/dev/null 2>&1; then
    git checkout -q $DWC -- tests/fpgadataflow/test_fpgadataflow_dwc.py 2>/dev/null || true
    # benchmark tests are replaced by the integration branch's afterwards anyway
    for f in $(git diff --name-only --diff-filter=U -- tests/benchmark setup.cfg); do
      git checkout -q $tests -- $f
    done
    git add -u
    if git diff --cached --name-only | xargs -r grep -l '^<<<<<<< ' ; then
      echo "unresolved conflicts merging $1"; exit 1
    fi
    git commit -q --no-edit
  fi
}

check_dwc() {
  local f=src/finn/custom_op/fpgadataflow/streamingdatawidthconverter.py
  if git merge-base --is-ancestor $DWC HEAD && git merge-base --is-ancestor $FIFO HEAD; then
    git checkout -q $GOOD_DWC_FILE -- $f
  git diff --cached --quiet || git commit -q -m "Merge fix: DWC cycles/tree model as on the integration branch"
    [ "$(grep -c 'def get_exp_cycles' $f)" = 1 ] || { echo "DWC invariant broken"; exit 1; }
    grep -q "if self.pads_or_crops():" $f || { echo "DWC tree-model guard missing"; exit 1; }
  fi
}

for t in "$@"; do
  case $t in
    integ) wt=$M/build/wt/integ; br=dev-finnexamples-features; base=$DWC; extra="$FIFO $PR $HLS $MVC $DEVPRS";;
    fifo) wt=$M/build/wt/fifo$S; br=test$S/fifo; base=$FIFO; extra="$PR $HLS $MVC $DEVPRS";;
    folding) wt=$M/build/wt/folding$S; br=test$S/folding; base=$FOLD; extra="$PR $FIFO $HLS $MVC $DEVPRS";;
    dwc) wt=$M/build/wt/dwc$S; br=test$S/dwc; base=$DWC; extra="$PR $FIFO $HLS $MVC $DEVPRS";;
    *) echo "unknown $t"; exit 1;;
  esac
  [ -d $wt ] || git -C $M worktree add -q $wt -b $br $base
  cd $wt
  # the tests are taken from the integration branch as it is before this rebuild
  tests=${TESTS_COMMIT:-$(git rev-parse dev-finnexamples-features)}
  git checkout -q -B $br $base
  for b in $extra; do merge $b; done
  check_dwc
  for c in $GAPREQ; do git merge-base --is-ancestor $c HEAD || git cherry-pick -x $c >/dev/null; done
  git checkout -q $tests -- tests/benchmark setup.cfg
  git rm -q --ignore-unmatch tests/benchmark/feature_flow_helpers.py \
    tests/benchmark/feature_flow_report.py tests/benchmark/test_fifo_sizing_flow.py \
    tests/benchmark/test_folding_flow.py tests/benchmark/test_dwc_padding_flow.py
  git diff --cached --quiet || git commit -q -m "Benchmark: per-model feature tests for all finn-examples models ($br)"
  $M/claude-tools/finn-examples-features/tools/wtsetup.sh $wt
  echo "$br: $(git log --oneline -1)"
done
