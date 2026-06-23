"""Linux Landlock sandbox: confine *writes* to a set of roots.

Reads and program execution are unaffected (normal Unix permissions apply);
only write-ish filesystem operations are denied outside the writable roots.
This is what stops the ``bash`` tool from modifying anything outside the
agent's workspace.

Linux-only. Needs the ``landlock`` package and a kernel with Landlock enabled
(Linux >= 5.13). Trimmed from the production sandbox in the parent repo to the
single sync code path this stub uses.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

libc = ctypes.CDLL("libc.so.6", use_errno=True)
PR_SET_NO_NEW_PRIVS = 38

# Grant the /dev directory so `>/dev/null` keeps working. Real device access is
# still gated by normal OS permissions, so this only exposes world-writable sinks.
_EXTRA_WRITABLE: tuple[str, ...] = ("/dev",)

# Every write-ish filesystem access we want to deny outside the writable roots.
# (READ_FILE / READ_DIR / EXECUTE are intentionally absent -> reads & running
# binaries stay unrestricted everywhere.)
_WRITE_FLAG_NAMES = (
    "WRITE_FILE",
    "TRUNCATE",
    "MAKE_REG",
    "MAKE_DIR",
    "MAKE_SYM",
    "MAKE_FIFO",
    "MAKE_SOCK",
    "MAKE_CHAR",
    "MAKE_BLOCK",
    "REMOVE_FILE",
    "REMOVE_DIR",
    "REFER",
)


@dataclass(frozen=True)
class SandboxConfig:
    writable_roots: Sequence[str]
    cwd: str | None = None
    # Optional resource limits (None = unlimited). Off by default so long
    # compiles/benchmarks are not killed; set them per call if you want caps.
    cpu_seconds: int | None = None
    as_bytes: int | None = None
    fsize_bytes: int | None = None
    umask: int = 0o022


def _no_new_privs() -> None:
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"prctl(PR_SET_NO_NEW_PRIVS) failed: {os.strerror(e)}")


def _set_rlimits(cfg: SandboxConfig) -> None:
    if cfg.cpu_seconds is not None:
        resource.setrlimit(resource.RLIMIT_CPU, (cfg.cpu_seconds, cfg.cpu_seconds))
    if cfg.as_bytes is not None:
        resource.setrlimit(resource.RLIMIT_AS, (cfg.as_bytes, cfg.as_bytes))
    if cfg.fsize_bytes is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (cfg.fsize_bytes, cfg.fsize_bytes))


def _write_mask():
    from landlock import FSAccess

    mask = FSAccess(0)
    for name in _WRITE_FLAG_NAMES:
        mask |= getattr(FSAccess, name)
    # Intersect with flags the running kernel actually supports (ABI version
    # determines the ceiling; e.g. REFER needs v2/5.19+, TRUNCATE needs v3/6.2+).
    return mask & FSAccess.all()


def _apply_sandbox(cfg: SandboxConfig) -> None:
    """Run in the child, right before exec: install Landlock + rlimits."""
    if sys.platform != "linux":
        raise RuntimeError("The sandbox is Linux-only.")

    from landlock import Ruleset

    _no_new_privs()
    _set_rlimits(cfg)

    mask = _write_mask()
    # Handle only write accesses -> they are denied everywhere they are not
    # explicitly allowed below. Reads remain unaffected.
    rs = Ruleset(restrict_rules=mask)
    for root in list(cfg.writable_roots) + list(_EXTRA_WRITABLE):
        rs.allow(root, rules=mask)
    rs.apply()

    os.umask(cfg.umask)


@contextlib.contextmanager
def sandbox_popen(
    args: Sequence[str], *, cfg: SandboxConfig, **popen_kwargs
) -> Iterator[subprocess.Popen]:
    """Like ``subprocess.Popen(args, ...)`` but the child can only write under
    ``cfg.writable_roots`` (+ /dev). Paths are resolved to absolute first."""
    roots = [str(Path(p).resolve()) for p in cfg.writable_roots]
    cwd = str(Path(cfg.cwd).resolve()) if cfg.cwd else None
    cfg = SandboxConfig(
        writable_roots=roots,
        cwd=cwd,
        cpu_seconds=cfg.cpu_seconds,
        as_bytes=cfg.as_bytes,
        fsize_bytes=cfg.fsize_bytes,
        umask=cfg.umask,
    )

    def _child_setup() -> None:
        _apply_sandbox(cfg)
        if cfg.cwd:
            os.chdir(cfg.cwd)

    proc = subprocess.Popen(
        list(args), preexec_fn=_child_setup, close_fds=True, **popen_kwargs
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.wait()


def run_sandboxed(
    argv: Sequence[str],
    *,
    writable_roots: Sequence[str],
    cwd: str,
    timeout: int = 120,
    output_limit: int = 20_000,
) -> str:
    """Run ``argv`` with writes confined to ``writable_roots``; return a string
    with the exit code and combined stdout/stderr. Shared by the bash and run
    tools so both behave identically."""
    cfg = SandboxConfig(writable_roots=list(writable_roots), cwd=cwd)
    with sandbox_popen(
        argv,
        cfg=cfg,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        try:
            out, _ = proc.communicate(timeout=timeout)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            return f"[timed out after {timeout}s]\n{out or ''}"

    out = out or ""
    if len(out) > output_limit:
        out = out[:output_limit] + f"\n... [truncated, {len(out)} chars total]"
    return f"(exit {code})\n{out}" if out.strip() else f"(exit {code}, no output)"
