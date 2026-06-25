"""Shared, cross-process progress reporting for the TAV optimization loops.

Several node-optimization loops may run at the same time (one OS process per
node -- see ``examples/tav_tree_model_loop.py`` multi-node mode), and they all
report into ONE shared table so a single file shows live progress for every
node. Writes are guarded by an advisory file lock (``fcntl.flock``) so the
concurrent processes never corrupt the file.

Delta ratio
-----------
For a model-produced token access vector ``got`` and the rtlsim reference
``ref``, the per-index delta ratio is ``|ref[i] - got[i]| / ref[i]``. We
aggregate the **max** and the **mean** of those ratios across every port of
every case in an evaluation and report them as percentages. Indices where
``ref[i] == 0`` are skipped (division by zero); a length mismatch counts each
extra/missing element as a full miss (ratio 1.0) so a wrong-length vector is
never scored as perfect.

The table rows are
    node_name   iteration   max_delta_ratio %   average_delta_ratio %
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

_HEADER = ("node_name", "iteration", "max_delta_ratio %", "average_delta_ratio %")


# ---------------------------------------------------------------------------
# delta ratio
# ---------------------------------------------------------------------------
def delta_ratios(result) -> tuple[float, float]:
    """(max, mean) per-index delta ratio over all ports/cases of an
    ``EvalResult``, expressed as percentages.

    ``ratio_i = |ref_i - got_i| / ref_i`` for every index where ``ref_i != 0``;
    a case that errored (no usable vector) and every extra/missing element of a
    length-mismatched vector count as a full miss (1.0) so neither is ever
    scored as a perfect match."""
    ratios: list[float] = []
    for c in getattr(result, "cases", []):
        if getattr(c, "error", None) is not None:
            ratios.append(1.0)
            continue
        for p in getattr(c, "ports", []):
            ref, got = p.ref, p.got
            n = min(len(ref), len(got))
            for i in range(n):
                r = ref[i]
                if r == 0:
                    continue
                ratios.append(abs(r - got[i]) / abs(r))
            ratios.extend([1.0] * abs(len(got) - len(ref)))
    if not ratios:
        return 0.0, 0.0
    return max(ratios) * 100.0, (sum(ratios) / len(ratios)) * 100.0


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
            f"{float(r.get('max_delta_ratio', 0.0)):.2f}",
            f"{float(r.get('avg_delta_ratio', 0.0)):.2f}",
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
    max_ratio: float,
    avg_ratio: float,
    *,
    table_path=None,
    data_path=None,
    extra: dict | None = None,
) -> dict:
    """Append one ``(node, iteration, max%, avg%)`` row to the shared progress
    table, locking across processes so concurrent node loops can't corrupt it.
    Returns the row dict that was recorded."""
    table_path = Path(table_path or DEFAULT_TABLE)
    data_path = Path(data_path or DEFAULT_DATA)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "node": node,
        "iteration": int(iteration),
        "max_delta_ratio": round(float(max_ratio), 4),
        "avg_delta_ratio": round(float(avg_ratio), 4),
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
