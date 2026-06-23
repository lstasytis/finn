"""apply_patch tool: apply a V4A-style patch, confined to the workspace.

Supported envelope::

    *** Begin Patch
    *** Add File: relative/path.txt
    +new file contents, line by line
    *** Update File: relative/path.txt
    @@ optional context anchor
     unchanged context line
    -removed line
    +added line
    *** Delete File: relative/path.txt
    *** End Patch

All paths are interpreted relative to the workspace and may not escape it --
that path check is the apply_patch counterpart to the bash sandbox (apply_patch
writes in-process, so Landlock on a subprocess would not cover it).
"""

from __future__ import annotations

import re
from pathlib import Path

# Lines the model sometimes emits when it forgets it's not writing a git diff.
# Silently dropped from *** Add File bodies (they carry no real content).
_GIT_DIFF_HEADER = re.compile(r"^(\+\+\+|---)\s|^@@.*@@")

APPLY_PATCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "apply_patch",
        "description": (
            "Create, update or delete files with a V4A patch. Wrap it in "
            "'*** Begin Patch' / '*** End Patch'. Use '*** Add File: <path>' "
            "followed by '+' lines for new files; '*** Update File: <path>' with "
            "context lines (leading space), '-' to remove and '+' to add, plus "
            "optional '@@ anchor' separators between hunks; '*** Delete File: "
            "<path>' to delete. Always include a few unchanged context lines "
            "around edits so the hunk can be located. Paths are relative to the "
            "workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "The full patch text."}
            },
            "required": ["patch"],
        },
    },
}


class PatchError(Exception):
    pass


def _safe_path(workspace: Path, rel: str) -> Path:
    p = (workspace / rel.strip()).resolve()
    if p != workspace and workspace not in p.parents:
        raise PatchError(f"Path {rel.strip()!r} escapes the workspace.")
    return p


def _ensure_trailing_newline(text: str) -> str:
    return text if (not text or text.endswith("\n")) else text + "\n"


def _apply_update(old_text: str, hunk_lines: list[str]) -> str:
    """Apply one Update-File body. Splits into sub-hunks at '@@' anchors and
    replaces each old block (context + removed) with its new block (context +
    added)."""
    groups: list[list[str]] = [[]]
    for ln in hunk_lines:
        if ln.startswith("@@"):
            groups.append([])
        else:
            groups[-1].append(ln)

    text = old_text
    for group in groups:
        old_chunk: list[str] = []
        new_chunk: list[str] = []
        for ln in group:
            tag = ln[0] if ln else " "
            rest = ln[1:] if ln else ""
            if tag == " ":
                old_chunk.append(rest)
                new_chunk.append(rest)
            elif tag == "-":
                old_chunk.append(rest)
            elif tag == "+":
                new_chunk.append(rest)
            else:
                raise PatchError(f"Bad patch line (expected ' ', '-' or '+'): {ln!r}")

        if not old_chunk and not new_chunk:
            continue

        old_block = "\n".join(old_chunk)
        new_block = "\n".join(new_chunk)
        if old_block:
            if old_block not in text:
                raise PatchError(
                    "Context not found in file; patch does not apply:\n" + old_block
                )
            text = text.replace(old_block, new_block, 1)
        else:
            # Pure addition with no context -> append at end of file.
            text = _ensure_trailing_newline(text) + new_block
    return text


def apply_patch(patch: str, workspace: Path) -> str:
    raw = patch.splitlines()
    if raw and raw[0].strip() == "*** Begin Patch":
        raw = raw[1:]
    # Drop trailing blank lines before checking for End Patch.
    # Without this, a model-appended trailing newline leaves raw[-1] == ""
    # and *** End Patch never gets stripped, leaking into the written file.
    while raw and not raw[-1].strip():
        raw.pop()
    if raw and raw[-1].strip() == "*** End Patch":
        raw = raw[:-1]

    results: list[str] = []
    i, n = 0, len(raw)
    while i < n:
        line = raw[i]
        if line.startswith("*** Add File:"):
            rel = line[len("*** Add File:") :]
            i += 1
            body: list[str] = []
            while i < n and not raw[i].startswith("*** "):
                body.append(raw[i])
                i += 1
            content = "\n".join(
                l[1:] if l.startswith("+") else l
                for l in body
                if not _GIT_DIFF_HEADER.match(l)
            )
            target = _safe_path(workspace, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_ensure_trailing_newline(content))
            results.append(f"added {rel.strip()}")
        elif line.startswith("*** Update File:"):
            rel = line[len("*** Update File:") :]
            i += 1
            body = []
            while i < n and not raw[i].startswith("*** "):
                body.append(raw[i])
                i += 1
            target = _safe_path(workspace, rel)
            if not target.exists():
                raise PatchError(f"Cannot update missing file: {rel.strip()}")
            target.write_text(_apply_update(target.read_text(), body))
            results.append(f"updated {rel.strip()}")
        elif line.startswith("*** Delete File:"):
            rel = line[len("*** Delete File:") :]
            i += 1
            target = _safe_path(workspace, rel)
            if target.exists():
                target.unlink()
            results.append(f"deleted {rel.strip()}")
        else:
            i += 1  # ignore stray lines

    if not results:
        raise PatchError("No file operations found in patch.")
    return "ok: " + "; ".join(results)
