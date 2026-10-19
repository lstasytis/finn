#!/bin/bash
# Link the untracked bits of the main worktree into a worktree: wtsetup.sh <worktree dir>
set -e
M=/home/lstasytis/finn
W=$(realpath "$1")
[ -e "$W/deps" ] || ln -s $M/deps "$W/deps"
grep -qxF '/deps' $M/.git/info/exclude || echo '/deps' >> $M/.git/info/exclude
ln -sf $M/finn_xsi/xsi.so "$W/finn_xsi/xsi.so"
mkdir -p "$W/tests/benchmark/verification_io"
for f in $M/tests/benchmark/models/*.onnx; do ln -sf "$f" "$W/tests/benchmark/models/"; done
for f in $M/tests/benchmark/verification_io/*.npy; do ln -sf "$f" "$W/tests/benchmark/verification_io/"; done
