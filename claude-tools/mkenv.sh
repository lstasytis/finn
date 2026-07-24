#!/usr/bin/env bash
# claude-tools/mkenv.sh — assemble a FINN working environment from layers.
#
#   env = BASE  +  claude-tools overlay  +  one or more feature branches
#
# The base is our modern dev; the claude-tools overlay is additive (never
# conflicts); feature branches carry the real work. Feature-vs-base and
# feature-vs-feature conflicts are auto-resolved by git rerere (pre-trained),
# so rebuilding an environment is cheap and repeatable.
#
# Usage:
#   claude-tools/mkenv.sh <env-branch> <feature-branch>...
#   claude-tools/mkenv.sh --worktree <dir> <env-branch> <feature-branch>...
#
# Examples:
#   claude-tools/mkenv.sh env/fifo  feature/analytical-fifo-sizing
#   claude-tools/mkenv.sh env/align feature/analytical-fifo-sizing feature/label_aligner-clean
#   claude-tools/mkenv.sh --worktree ../finn-align env/align \
#        feature/analytical-fifo-sizing feature/label_aligner-clean
#
# Override the base / overlay branch names via env vars:
#   MKENV_BASE=feature/expanded-finn-examples  MKENV_TOOLS=claude-tools
set -euo pipefail

# Re-exec from a stable temp copy: mkenv briefly checks out the base branch,
# which does NOT contain claude-tools/, so the in-tree copy of this script can
# vanish mid-run. Running from a copy makes "run it from any branch" always safe.
if [ -z "${MKENV_REEXEC:-}" ]; then
  _self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  _tmp="$(mktemp)"; cat "$_self" > "$_tmp"
  MKENV_REEXEC=1 exec bash "$_tmp" "$@"
fi

BASE="${MKENV_BASE:-feature/expanded-finn-examples}"
TOOLS="${MKENV_TOOLS:-claude-tools}"

WORKTREE=""
if [ "${1:-}" = "--worktree" ]; then WORKTREE="$2"; shift 2; fi
[ $# -ge 2 ] || { echo "usage: mkenv.sh [--worktree DIR] <env-branch> <feature>..." >&2; exit 2; }
ENV="$1"; shift
FEATURES=("$@")

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Refuse to run on a dirty tree (tracked changes) -- git switch -C would carry or
# clobber them. Untracked scratch files (*.sv, build dirs) are fine and ignored.
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "!! working tree has uncommitted tracked changes; commit or stash first:" >&2
  git status --short --untracked-files=no | sed 's/^/     /' >&2
  exit 1
fi

# rerere = "reuse recorded resolution": remembers how a conflict was resolved
# and replays it automatically the next time the same conflict appears.
git config rerere.enabled true
git config rerere.autoupdate true

echo ">> assembling '$ENV' = $BASE + $TOOLS + ${FEATURES[*]}"

# (Re)create the env branch at BASE, optionally checked out in its own worktree
# so several environments can live side by side.
if [ -n "$WORKTREE" ]; then
  git worktree remove --force "$WORKTREE" 2>/dev/null || true
  git branch -f "$ENV" "$BASE"
  git worktree add "$WORKTREE" "$ENV"
  cd "$WORKTREE"
else
  git switch -C "$ENV" "$BASE"
fi

merge() {  # merge $1; if rerere fully resolved the conflicts, commit and go on
  local ref="$1"
  if git merge --no-edit "$ref"; then return 0; fi
  if [ -z "$(git diff --name-only --diff-filter=U)" ]; then
    git commit --no-edit --no-verify
    echo "   (rerere auto-resolved conflicts from $ref)"
    return 0
  fi
  echo "!! unresolved conflict while merging $ref:" >&2
  git diff --name-only --diff-filter=U | sed 's/^/     /' >&2
  echo "   Resolve them, then: git add -A && git commit --no-edit" >&2
  echo "   rerere records your resolution, so next mkenv run is automatic." >&2
  exit 1
}

merge "$TOOLS"
for f in "${FEATURES[@]}"; do merge "$f"; done

# Install our finn-hlslib additions (AlignLabels + activations.hpp) into the
# gitignored clone; harmless if deps/finn-hlslib isn't populated yet.
if [ -d "$(git rev-parse --show-toplevel)/deps/finn-hlslib" ]; then
  bash "$(git rev-parse --show-toplevel)/claude-tools/patch_hlslib.sh" || true
fi

echo ">> '$ENV' ready${WORKTREE:+ in $WORKTREE}"
echo "   layers: $BASE -> $TOOLS -> ${FEATURES[*]}"
