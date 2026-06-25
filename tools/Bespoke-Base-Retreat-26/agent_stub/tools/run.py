"""run tool: execute a program the agent produced, confined to the workspace.

Unlike `bash` (which also gets /tmp as scratch), `run` confines the executed
program's writes to *only* the workspace/output folder -- it is meant for
running the artifact the agent just wrote, with the tightest sandbox.
"""

from __future__ import annotations

from pathlib import Path

from agent_stub.sandbox import run_sandboxed

# Map file extension -> interpreter. Unknown extensions fall back to executing
# the file directly (it must have an executable bit + shebang).
_INTERPRETERS: dict[str, list[str]] = {
    ".py": ["python3"],
    ".sh": ["bash"],
    ".js": ["node"],
}

RUN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run",
        "description": (
            "Run a program file you created in the workspace (e.g. a Python "
            "script) -- for your own scratch use (quick checks, computing a "
            "value, debugging a snippet). It does not evaluate your actual "
            "deliverable; that happens automatically, on its own, between "
            "iterations. Give the path relative to the workspace and optional "
            "CLI args. The program is sandboxed: it may only write inside the "
            "workspace/output folder. Returns the exit code and combined "
            "stdout/stderr. Interpreter is chosen by extension (.py, .sh, .js)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Program file path, relative to the workspace (e.g. 'game.py').",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional command-line arguments for the program.",
                },
            },
            "required": ["path"],
        },
    },
}


def run_program(
    path: str,
    workspace: Path,
    args: list[str] | None = None,
    timeout: int = 120,
) -> str:
    target = (workspace / path.strip()).resolve()
    if target != workspace and workspace not in target.parents:
        return f"error: path {path.strip()!r} escapes the workspace."
    if not target.is_file():
        return f"error: no such file in workspace: {path.strip()!r}"

    argv = _INTERPRETERS.get(target.suffix, []) + [str(target)]
    argv += [str(a) for a in (args or [])]

    # Confined to ONLY the workspace/output folder (no /tmp).
    return run_sandboxed(
        argv,
        writable_roots=[str(workspace)],
        cwd=str(workspace),
        timeout=timeout,
    )
