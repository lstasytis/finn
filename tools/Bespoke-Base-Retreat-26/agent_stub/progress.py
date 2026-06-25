"""Shared, cross-process progress reporting for the TAV optimization loops.

Several node-optimization loops may run at the same time (one OS process per
node -- see ``examples/tav_tree_model_loop.py`` multi-node mode), and they all
report into ONE shared table so a single file shows live progress for every
node. Writes are guarded by an advisory file lock (``fcntl.flock``) so the
concurrent processes never corrupt the file.

Delta metrics
-------------
For a model-produced token access vector ``got`` and the rtlsim reference
``ref``, the per-index absolute delta is ``|ref[i] - got[i]|`` and the per-index
normalized delta is ``|ref[i] - got[i]| / ref[i]``. Across every port of every
case in an evaluation we report two numbers:

  * ``max_abs_delta`` -- the single worst absolute delta (a raw token-count
    difference, not a percentage).
  * ``average_normalized_delta %`` -- the mean of the normalized deltas, as a
    percentage. Indices where ``ref[i] == 0`` are skipped for normalization; a
    length mismatch counts each extra/missing element as a full miss (1.0) so a
    wrong-length vector is never scored as perfect.

The table rows are
    node_name   iteration   max_abs_delta   average_normalized_delta %
appended once per iteration of any node. A JSONL sidecar is the source of truth;
the aligned ``.txt`` table is re-rendered from it on every update.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:  # POSIX advisory locking; the loops only run on Linux (Landlock anyway).
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX host
    fcntl = None

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_TABLE = _OUTPUTS_DIR / "progress_table.txt"
DEFAULT_DATA = _OUTPUTS_DIR / "progress.jsonl"

_HEADER = ("node_name", "iteration", "max_abs_delta", "average_normalized_delta %")


# ---------------------------------------------------------------------------
# delta metrics
# ---------------------------------------------------------------------------
def delta_ratios(result) -> tuple[float, float]:
    """(worst absolute delta, average normalized delta %) over all ports/cases.

    The first value is the single largest ``|ref_i - got_i|`` across every port
    of every case -- a raw token-count difference, NOT a percentage. The second
    is the mean of ``|ref_i - got_i| / ref_i`` (indices where ``ref_i == 0`` are
    skipped) as a percentage. A case that errored, and every extra/missing
    element of a length-mismatched vector, counts as a full normalized miss
    (1.0); a length mismatch also contributes the unmatched tail value to the
    absolute delta, so a wrong-length vector is never scored as perfect."""
    abs_deltas: list[float] = []
    norm: list[float] = []
    for c in getattr(result, "cases", []):
        if getattr(c, "error", None) is not None:
            norm.append(1.0)
            continue
        for p in getattr(c, "ports", []):
            ref, got = p.ref, p.got
            n = min(len(ref), len(got))
            for i in range(n):
                d = abs(ref[i] - got[i])
                abs_deltas.append(d)
                if ref[i] != 0:
                    norm.append(d / abs(ref[i]))
            # length mismatch: each extra/missing element is a full normalized
            # miss, and its (cumulative) value is an absolute delta vs nothing.
            longer = got if len(got) > len(ref) else ref
            for i in range(n, len(longer)):
                abs_deltas.append(abs(longer[i]))
                norm.append(1.0)
    max_abs = max(abs_deltas) if abs_deltas else 0.0
    avg_norm = (sum(norm) / len(norm)) * 100.0 if norm else 0.0
    return float(max_abs), avg_norm


# ---------------------------------------------------------------------------
# shared table
# ---------------------------------------------------------------------------
def _load_rows(data_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(data_path):
        return rows
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _render(table_path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r.get("node", ""), r.get("iteration", 0), r.get("ts", 0)))
    cells = [
        (
            str(r.get("node", "")),
            str(r.get("iteration", "")),
            f"{float(r.get('max_abs_delta', 0.0)):g}",
            f"{float(r.get('avg_normalized_delta_pct', 0.0)):.2f}",
        )
        for r in rows
    ]
    widths = [len(h) for h in _HEADER]
    for row in cells:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt(vals):
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(vals))

    lines = [fmt(_HEADER), fmt(["-" * w for w in widths])]
    lines += [fmt(row) for row in cells]
    tmp = Path(str(table_path) + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, table_path)  # atomic swap so a concurrent reader never sees half a file


def record_iteration(
    node: str,
    iteration: int,
    max_abs_delta: float,
    avg_normalized_delta_pct: float,
    *,
    table_path=None,
    data_path=None,
    extra: dict | None = None,
) -> dict:
    """Append one ``(node, iteration, max_abs_delta, avg_normalized_delta %)``
    row to the shared progress table, locking across processes so concurrent
    node loops can't corrupt it. Returns the row dict that was recorded."""
    table_path = Path(table_path or DEFAULT_TABLE)
    data_path = Path(data_path or DEFAULT_DATA)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "node": node,
        "iteration": int(iteration),
        "max_abs_delta": round(float(max_abs_delta), 4),
        "avg_normalized_delta_pct": round(float(avg_normalized_delta_pct), 4),
        "ts": time.time(),
    }
    if extra:
        row.update(extra)

    lock_path = Path(str(table_path) + ".lock")
    with open(lock_path, "w") as lf:
        if fcntl is not None:
            fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            with open(data_path, "a") as df:
                df.write(json.dumps(row) + "\n")
            _render(table_path, _load_rows(data_path))
        finally:
            if fcntl is not None:
                fcntl.flock(lf, fcntl.LOCK_UN)
    return row
