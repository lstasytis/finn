#!/usr/bin/env bash
# claude-tools/patch_hlslib.sh — install our finn-hlslib additions into the clone.
#
# deps/finn-hlslib is gitignored and provisioned by fetch-repos.sh; NOT tracked in
# git (tracking dep files breaks on branch switching -- a branch that doesn't track
# them deletes them from the shared clone). Instead we keep complete reference
# copies under claude-tools/hlslib/ and copy them into the clone here:
#   * streamtools.h  -- vanilla (pinned commit) + the AlignLabels template
#   * activations.hpp -- the clone shipped incomplete without it; MVAU/thresholding
#                        HLS #include it, so synthesis needs it present
#
# Run after fetch-repos.sh (or after assembling an env with mkenv.sh, which calls
# this automatically). Idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
DEST="$ROOT/deps/finn-hlslib"

if [ ! -d "$DEST" ]; then
  echo "!! $DEST not found -- run fetch-repos.sh first" >&2
  exit 1
fi

for f in streamtools.h activations.hpp; do
  cp "$HERE/hlslib/$f" "$DEST/$f"
  echo "installed deps/finn-hlslib/$f"
done
echo ">> finn-hlslib patched (AlignLabels + activations.hpp present)"
