#!/bin/bash
# Restart the FINN dev container (Vivado 2024.2) after the host reboot and resume the
# Claude session that ran in it. Run on the HOST, from /home/lstasytis/finn.
#
#   ./restart.sh save    BEFORE the reboot, while the container still runs: snapshots the
#                        whole container (incl. $HOME=/tmp/home_dir with Claude's login,
#                        session history and pip --user packages) into a local image.
#   ./restart.sh         AFTER the reboot: starts the container from that snapshot (or, if
#                        there is none, from the original image; then restore $HOME inside
#                        the container with the command printed below).
#
# Inside the new container: `claude --continue` (in /home/lstasytis/finn) resumes the session;
# the agent picks up from PROGRESS.md.
#
# Persistent anyway (host mounts): the repo /home/lstasytis/finn (incl. build/finn_build, the
# worktrees in build/wt, the lint env .lintenv), /tmp/finn_dev_lstasytis, /mnt/labstore/Xilinx.
# Lost without a snapshot: everything else in the container, notably /tmp/home_dir; a tarball of
# it is kept in build/home_backup/ as a fallback (written by the agent before the reboot).
set -e
cd "$(dirname "$(readlink -f "$0")")"
REPO=$(pwd)
NAME=finn_dev_${USER}
SNAP=finn-resume:latest
STATE=$REPO/build/home_backup/restart.env
mkdir -p "$REPO/build/home_backup"

find_container() {
  docker ps -q | xargs -r docker inspect -f '{{.Id}} {{.Config.Hostname}}' |
    awk -v n="$NAME" '$2 == n {print $1}' | head -1
}

if [ "$1" = "save" ]; then
  cid=$(find_container)
  [ -n "$cid" ] || { echo "no running container with hostname $NAME"; exit 1; }
  base=$(docker inspect -f '{{.Config.Image}}' "$cid")
  echo "BASE_IMAGE=$base" > "$STATE"
  echo "snapshotting container $cid (image $base) into $SNAP ..."
  docker commit -m "FINN dev container state before reboot $(date -I)" "$cid" "$SNAP"
  echo "SNAP_IMAGE=$SNAP" >> "$STATE"
  echo "done. After the reboot run: ./restart.sh"
  exit 0
fi

[ -f "$STATE" ] && source "$STATE"
if docker image inspect "$SNAP" > /dev/null 2>&1; then
  TAG=$SNAP
  RESTORE=""
else
  TAG=${BASE_IMAGE:-$(docker images xilinx/finn --format '{{.Repository}}:{{.Tag}}' | head -1)}
  RESTORE="tar -xpf $REPO/build/home_backup/claude_home.tar -C /tmp/home_dir"
fi
echo "starting $TAG"
[ -n "$RESTORE" ] && echo "no snapshot: inside the container first run:  $RESTORE"
echo "then: cd $REPO && claude --continue"

# same settings as the original launch; add your usual extras via FINN_DOCKER_EXTRA etc.
export FINN_XILINX_PATH=/mnt/labstore/Xilinx
export FINN_XILINX_VERSION=2024.2
export FINN_SKIP_DEP_REPOS=1      # keep deps/ as is
export FINN_DOCKER_PREBUILT=1     # never rebuild the image
export FINN_DOCKER_TAG=$TAG
exec ./run-docker.sh "$@"
