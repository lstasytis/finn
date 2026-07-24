#!/usr/bin/env bash
# claude-tools/whose.sh — which feature branch owns each file?
#
# A feature branch "owns" the files in its diff vs the base. Use this in an env to
# find out where a change belongs before committing it.
#
# Usage: claude-tools/whose.sh <file>...
#        claude-tools/whose.sh            # all currently-modified tracked files
set -euo pipefail
BASE="${MKENV_BASE:-feature/expanded-finn-examples}"
ROOT="$(git rev-parse --show-toplevel)"; cd "$ROOT"

FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then mapfile -t FILES < <(git diff --name-only HEAD); fi
[ ${#FILES[@]} -gt 0 ] || { echo "no files given and nothing modified" >&2; exit 1; }

mapfile -t FEATS < <(git for-each-ref --format='%(refname:short)' refs/heads/feature/ | grep -vx "$BASE" || true)

for f in "${FILES[@]}"; do
  owners=()
  for b in "${FEATS[@]}"; do
    if git diff --name-only "$BASE".."$b" -- | grep -qxF "$f"; then owners+=("$b"); fi
  done
  if [ ${#owners[@]} -eq 0 ]; then
    echo "$f  ->  (none: base file or brand-new) — decide manually"
  else
    echo "$f  ->  ${owners[*]}"
  fi
done
