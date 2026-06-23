"""Multi-step prompt pipeline with interactive control and persistent history.

Define your stages upfront, run them one by one. The full LLM conversation
context carries forward across every step. State is logged after each stage —
use --resume to pick up exactly where you left off.

Load stages from a directory (recommended):
    python pipeline.py --prompts-dir prompts/calc_challenge
    python pipeline.py --prompts-dir prompts/calc_challenge --auto-continue

Define stages inline or from a text file:
    python pipeline.py --prompts "write primes.py" "add unit tests"
    python pipeline.py --prompts-file plan.txt

Resume from a saved log:
    python pipeline.py --resume /mnt/labstore/Retreat-logs/20260619_123456_gemma.json
    python pipeline.py --resume /mnt/labstore/Retreat-logs/... --extend "one more step"

Prompts-dir format — each stage is a numbered .toml file:
    prompts/my_task/
      01_build.toml       name = "build"  max_turns = 20  prompt = \"""...\"""
      02_test.toml        name = "test"   max_turns = 15  prompt = \"""...\"""

Prompts-file format (blank line separates stages, global --max-turns applies):
    Write a Python calculator.

    Add input validation and run it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from agent_stub.models import DEFAULT_MODEL
from agent_stub.pipeline import DEFAULT_MAX_TURNS, LOG_DIR, Pipeline, Stage, load_stages_from_dir


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description="Run a multi-step agent pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prompts-dir", metavar="DIR",
        help="Directory of numbered .toml stage files (01_name.toml, 02_name.toml, ...).",
    )
    source.add_argument(
        "--prompts", nargs="+", metavar="PROMPT",
        help="One or more prompts as inline arguments (uses global --max-turns).",
    )
    source.add_argument(
        "--prompts-file", metavar="FILE",
        help="Text file of prompts; blank lines separate stages (uses global --max-turns).",
    )
    source.add_argument(
        "--resume", metavar="LOG",
        help="Path to a saved log file; continues from where it stopped.",
    )

    ap.add_argument(
        "--extend", nargs="+", metavar="PROMPT",
        help="With --resume: append extra stages using global --max-turns.",
    )
    ap.add_argument(
        "--name", default="",
        help="Short label embedded in the log filename (e.g. 'btree', 'my-experiment')."
             " Defaults to the --prompts-dir basename.",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--log-dir", default=str(LOG_DIR), metavar="DIR",
                    help=f"Where to write log files (default: {LOG_DIR}).")
    ap.add_argument(
        "--auto-continue", action="store_true",
        help="Skip the [c/i/a] menu and run all stages automatically.",
    )
    ap.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help="Default max turns per stage (overridden by per-stage setting in .toml).",
    )
    args = ap.parse_args()

    if args.resume:
        pipeline = Pipeline.load(Path(args.resume))
        if args.extend:
            extra = [Stage(prompt=p, max_turns=args.max_turns) for p in args.extend]
            pipeline.stages.extend(extra)
            pipeline.save()
        name_tag = f"  [{pipeline.name}]" if pipeline.name else ""
        print(f"Resuming{name_tag}  {len(pipeline.stages_completed)} step(s) done, "
              f"{len(pipeline.stages)} remaining")
        print(f"log → {pipeline.log_path}  (continuing in same file)")
        if args.workspace != "workspace":
            pipeline.workspace = Path(args.workspace).resolve()
    else:
        # Derive pipeline name: explicit --name wins, then --prompts-dir basename
        name = args.name or (Path(args.prompts_dir).name if args.prompts_dir else "")

        if args.prompts_dir:
            stages = load_stages_from_dir(Path(args.prompts_dir), args.max_turns)
        elif args.prompts_file:
            stages = [Stage(prompt=p, max_turns=args.max_turns)
                      for p in _load_prompts_file(Path(args.prompts_file))]
        else:
            stages = [Stage(prompt=p, max_turns=args.max_turns) for p in args.prompts]

        pipeline = Pipeline.new(
            stages=stages,
            model=args.model,
            workspace=Path(args.workspace).resolve(),
            log_dir=Path(args.log_dir),
            name=name,
        )
        print(f"log → {pipeline.log_path}")
        for i, s in enumerate(pipeline.stages, 1):
            label = f"  [{s.name}]" if s.name else ""
            print(f"  {i}.{label} max_turns={s.max_turns}  "
                  f"{s.prompt[:60]}{'...' if len(s.prompt) > 60 else ''}")

    pipeline.run(auto_continue=args.auto_continue)


def _load_prompts_file(path: Path) -> list[str]:
    """Split on blank lines; each paragraph becomes one stage prompt."""
    text = path.read_text()
    blocks = [b.strip() for b in text.split("\n\n")]
    return [b for b in blocks if b]


if __name__ == "__main__":
    main()
