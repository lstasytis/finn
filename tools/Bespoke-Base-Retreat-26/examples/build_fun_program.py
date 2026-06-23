"""Example: let the agent build (and run) a small, fun program in the workspace.

The agent writes files with apply_patch and runs them with bash -- all confined
to the workspace by the sandbox.

Run (default local model on dgx02):
    python examples/build_fun_program.py
    python examples/build_fun_program.py --model unsloth/DeepSeek-V4-Flash
    python examples/build_fun_program.py --task "your own task here"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent_stub.agent import run_agent
from agent_stub.models import DEFAULT_MODEL

DEFAULT_TASK = (
    "Create `fortune.py`: a colorful terminal program that prints a random "
    "programming fortune cookie inside a hand-drawn ASCII speech bubble, with a "
    "little ASCII cookie underneath. Use ANSI color codes. Include at least 8 "
    "witty, varied fortunes and pick one at random on each run. Then run it three "
    "times to show three different fortunes."
)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Have the agent build a fun program.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--max-turns", type=int, default=30)
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    print(f"model={args.model}  workspace={workspace}\n")
    run_agent(args.task, args.model, workspace, max_turns=args.max_turns)


if __name__ == "__main__":
    main()
