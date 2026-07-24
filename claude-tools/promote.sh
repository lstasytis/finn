#!/usr/bin/env bash
# claude-tools/promote.sh — land an uncommitted env change onto the feature branch
# that owns it, so it survives env rebuilds (env/* branches are disposable).
#
# It refuses to promote files the target feature does not own (they belong to a
# different feature) -- run `whose.sh` to see where they go, and promote per branch.
#
# Usage:
#   claude-tools/promote.sh <feature-branch> -m "message" [file ...]
#   (no files -> all currently-modified tracked files)
set -euo pipefail
BASE="${MKENV_BASE:-feature/expanded-finn-examples}"
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"

[ $# -ge 1 ] || { echo "usage: promote.sh <feature-branch> -m msg [file...]" >&2; exit 2; }
FEATURE="$1"; shift
MSG="promoted from env"
if [ "${1:-}" = "-m" ]; then MSG="$2"; shift 2; fi
FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then mapfile -t FILES < <(git diff --name-only HEAD); fi
[ ${#FILES[@]} -gt 0 ] || { echo "no modified files to promote" >&2; exit 1; }

git rev-parse --verify --quiet "$FEATURE" >/dev/null || { echo "no such branch: $FEATURE" >&2; exit 1; }

# Ownership guard: every file must be one the target feature actually touches vs
# base (or a brand-new file owned by nobody). Files owned by a *different* feature
# are rejected so a change never lands on the wrong branch.
mapfile -t OWNED < <(git diff --name-only "$BASE".."$FEATURE" --)
owned_by_target(){ printf '%s\n' "${OWNED[@]}" | grep -qxF "$1"; }
owned_elsewhere(){  # is $1 owned by some OTHER feature branch?
  local b
  for b in $(git for-each-ref --format='%(refname:short)' refs/heads/feature/ | grep -vx "$BASE"); do
    [ "$b" = "$FEATURE" ] && continue
    git diff --name-only "$BASE".."$b" -- | grep -qxF "$1" && { echo "$b"; return 0; }
  done
  return 1
}
BAD=0
for f in "${FILES[@]}"; do
  if owned_by_target "$f"; then continue; fi
  if other="$(owned_elsewhere "$f")"; then
    echo "!! $f is owned by $other, not $FEATURE" >&2; BAD=1
  else
    echo ".. $f is new/unowned; will be added to $FEATURE" >&2
  fi
done
[ $BAD -eq 0 ] || { echo "refusing: split the promote per owning branch (see whose.sh)" >&2; exit 1; }

PATCH="$(mktemp)"; git diff HEAD -- "${FILES[@]}" > "$PATCH"
[ -s "$PATCH" ] || { echo "no diff in the given files" >&2; exit 1; }

WT="$(mktemp -d)"
git worktree add --quiet "$WT" "$FEATURE"
( cd "$WT"
  git apply --3way "$PATCH"
  git add -A
  git commit --no-verify -m "$MSG"
)
git worktree remove --force "$WT"
echo ">> promoted ${#FILES[@]} file(s) to $FEATURE: $MSG"
echo "   (the env still has the change; rebuild envs with mkenv to re-derive from $FEATURE)"
