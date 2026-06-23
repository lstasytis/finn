"""bash tool: run a shell command, confined by the Landlock sandbox so it can
only write inside the workspace (and /tmp). Reads are unrestricted."""

from __future__ import annotations

from pathlib import Path

from agent_stub.sandbox import run_sandboxed

# OpenAI-style tool schema, used by both cloud and vLLM endpoints.
BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a bash command in the workspace directory. Writes are sandboxed "
            "to the workspace (and /tmp); reads and program execution are "
            "unrestricted. Returns combined stdout/stderr and the exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
            },
            "required": ["command"],
        },
    },
}

def run_bash(command: str, workspace: Path, timeout: int = 120) -> str:
    # bash may write the workspace and /tmp (scratch).
    return run_sandboxed(
        ["/bin/sh", "-c", command],
        writable_roots=[str(workspace), "/tmp"],
        cwd=str(workspace),
        timeout=timeout,
    )
