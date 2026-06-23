"""CLI entry point for the retreat agent stub.

Examples:
    python main.py "create hello.py that prints hi, then run it"
    python main.py --model gpt-5.1 "refactor utils.py"
    echo "fix the failing test" | python main.py --model unsloth/DeepSeek-V4-Flash
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent_stub.agent import run_agent
from agent_stub.models import DEFAULT_MODEL, MODELS


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Minimal sandboxed LLM coding agent (OpenAI + local vLLM)."
    )
    ap.add_argument(
        "task", nargs="?", help="Task for the agent. If omitted, read from stdin."
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS),
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--workspace",
        default="workspace",
        help="Writable workspace dir -- the only place the agent may write (default: ./workspace).",
    )
    ap.add_argument("--max-turns", type=int, default=30)
    args = ap.parse_args()

    task = args.task if args.task is not None else sys.stdin.read()
    if not task.strip():
        ap.error("No task provided (pass it as an argument or pipe it on stdin).")

    workspace = Path(args.workspace).resolve()
    print(f"model={args.model}  workspace={workspace}")
    run_agent(task, args.model, workspace, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
