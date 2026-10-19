# Restarting the FINN container (Vivado 2024.2) without reinstalling

Why these steps: `$HOME` in the container (`/tmp/home_dir`) is **not** on a host mount. It
holds the Claude Code binary, the login and the session history, so a restart wipes it.
The repo (`/home/lstasytis/finn_dev_lstasytis/finn`) and `$FINN_BUILD_DIR`
(`/tmp/finn_dev_lstasytis`) are host mounts and survive the restart.

Before you start: make sure nothing is running in the container (`pgrep -af "vivado|java|pytest"`
prints nothing).

## 1. Inside the container: back up Claude's state

Quit Claude first (`/exit`), so the session history is complete, then:

```
mkdir -p /tmp/finn_dev_lstasytis/home_backup
tar -cpf /tmp/finn_dev_lstasytis/home_backup/claude_home.tar -C /tmp/home_dir \
    .claude .claude.json .local/share/claude .local/bin/claude .local/state/claude
exit
```

Leaving the shell stops the container. It was started with `--rm`, so it is also removed.

## 2. On the host (in your screen): find the existing image

```
docker images xilinx/finn --format '{{.Repository}}:{{.Tag}}  {{.CreatedSince}}'
```

Take the most recent tag. Pin it in step 3: otherwise `run-docker.sh` computes a new tag
from `git describe --dirty` (new commits, new tag) and builds a new image.

## 3. On the host: relaunch with Vivado 2024.2

```
cd /home/lstasytis/finn_dev_lstasytis/finn
FINN_XILINX_PATH=/mnt/labstore/Xilinx FINN_XILINX_VERSION=2024.2 \
FINN_SKIP_DEP_REPOS=1 FINN_DOCKER_PREBUILT=1 FINN_DOCKER_TAG=<tag from step 2> \
./run-docker.sh
```

Add any other variables you normally pass (`FINN_DOCKER_EXTRA`, a different
`FINN_HOST_BUILD_DIR`, ...).

* `FINN_SKIP_DEP_REPOS=1`: skips `fetch-repos.sh`, so the `deps/` checkouts (DynaRapid and
  its patches) stay as they are.
* `FINN_DOCKER_PREBUILT=1`: uses the existing image instead of running `docker build`.
* Still runs at startup (expected, harmless): the entrypoint's `pip install -e` steps, and a
  one-time DynaRapid recompile (~1 min) because `docker/dynarapid/*.patch` is newer than
  the last build.

## 4. Inside the new container: restore Claude and resume

```
tar -xpf /tmp/finn_dev_lstasytis/home_backup/claude_home.tar -C /tmp/home_dir
echo $XILINX_VIVADO     # expect /mnt/labstore/Xilinx/Vivado/2024.2
claude --continue
```

## Optional: git identity

`$HOME` is reset on every start, so there is no global git identity. Repo-local config
lives on the host and survives restarts:

```
git config user.name "Lukas Stasytis"
git config user.email "<your email>"
```
