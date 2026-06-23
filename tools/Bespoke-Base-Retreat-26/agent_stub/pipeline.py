"""Multi-step pipeline: run a sequence of stages with shared message history.

State is written to a JSON log after every stage. The log contains the full
LLM message chain, so loading it reconstructs the exact conversation context
and the run can continue as if it never stopped.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from agent_stub.agent import SYSTEM_PROMPT, run_agent_with_history

LOG_DIR = Path("/mnt/labstore/Retreat-logs")
DEFAULT_MAX_TURNS = 30


@dataclass
class Stage:
    prompt: str
    max_turns: int = DEFAULT_MAX_TURNS
    name: str = ""

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "max_turns": self.max_turns, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict | str) -> "Stage":
        if isinstance(d, str):  # backward compat: old logs stored plain strings
            return cls(prompt=d)
        return cls(
            prompt=d["prompt"],
            max_turns=d.get("max_turns", DEFAULT_MAX_TURNS),
            name=d.get("name", ""),
        )


def load_stages_from_dir(path: Path, default_max_turns: int = DEFAULT_MAX_TURNS) -> list[Stage]:
    """Load stages from a directory of .toml files, sorted by filename.

    Each .toml file must have a `prompt` key (multi-line string).
    Optional keys: `name` (str), `max_turns` (int).

    Example stage file (01_implement.toml):
        name = "implement"
        max_turns = 20
        prompt = \"""
        Write calc.py containing ...
        \"""
    """
    files = sorted(path.glob("*.toml"))
    if not files:
        raise FileNotFoundError(f"No .toml stage files found in {path}")
    stages = []
    for f in files:
        data = tomllib.loads(f.read_text())
        if "prompt" not in data:
            raise KeyError(f"{f}: missing required 'prompt' key")
        stages.append(Stage(
            prompt=data["prompt"].strip(),
            max_turns=data.get("max_turns", default_max_turns),
            name=data.get("name", f.stem),
        ))
    return stages


class Pipeline:
    def __init__(
        self,
        stages: list[Stage],
        model: str,
        workspace: Path,
        log_path: Path,
        messages: list[dict] | None = None,
        stages_completed: list[Stage] | None = None,
        created_at: str | None = None,
        name: str = "",
    ):
        self.model = model
        self.workspace = workspace
        self.log_path = log_path
        self.stages = list(stages)
        self.stages_completed: list[Stage] = stages_completed or []
        self.messages: list[dict] = messages or [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()
        self.name = name

    @classmethod
    def new(
        cls,
        stages: list[Stage],
        model: str,
        workspace: Path,
        log_dir: Path = LOG_DIR,
        name: str = "",
    ) -> "Pipeline":
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        user = _safe_slug(os.getenv("USER") or getpass.getuser(), maxlen=16)
        model_tag = _safe_slug(model.split("/")[-1], maxlen=20)
        parts = [ts, user]
        if name:
            parts.append(_safe_slug(name, maxlen=24))
        parts.append(model_tag)
        log_path = log_dir / f"{'_'.join(parts)}.json"
        return cls(stages, model, workspace, log_path, name=name)

    @classmethod
    def load(cls, log_path: Path) -> "Pipeline":
        data = json.loads(log_path.read_text())
        # Support old format (prompts_*) and new format (stages_*)
        remaining = data.get("stages_remaining", data.get("prompts_remaining", []))
        completed = data.get("stages_completed", data.get("prompts_completed", []))
        return cls(
            stages=[Stage.from_dict(s) for s in remaining],
            model=data["model"],
            workspace=Path(data["workspace"]),
            log_path=log_path,
            messages=data["messages"],
            stages_completed=[Stage.from_dict(s) for s in completed],
            created_at=data.get("created_at"),
            name=data.get("name", ""),
        )

    def save(self) -> None:
        payload = {
            "name": self.name,
            "model": self.model,
            "workspace": str(self.workspace),
            "created_at": self.created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "stages_completed": [s.to_dict() for s in self.stages_completed],
            "stages_remaining": [s.to_dict() for s in self.stages],
            "messages": self.messages,
        }
        self.log_path.write_text(json.dumps(payload, indent=2))

    def run(self, auto_continue: bool = False) -> None:
        total = len(self.stages_completed) + len(self.stages)

        while self.stages:
            stage = self.stages[0]
            step = len(self.stages_completed) + 1
            label = f"  [{stage.name}]" if stage.name else ""

            print(f"\n{'='*60}")
            print(f"step {step}/{total}{label}  max_turns={stage.max_turns}")
            print(f"  {stage.prompt[:100]}{'...' if len(stage.prompt) > 100 else ''}")
            print("=" * 60)

            self.messages.append({"role": "user", "content": stage.prompt})
            _, self.messages = run_agent_with_history(
                self.messages, self.model, self.workspace, max_turns=stage.max_turns
            )
            self.stages.pop(0)
            self.stages_completed.append(stage)
            self.save()
            print(f"\nsaved → {self.log_path}")

            if not self.stages:
                print("\nAll steps complete.")
                break

            if auto_continue:
                continue

            action = _menu()
            if action == "a":
                print("Aborted.")
                break
            if action == "i":
                text = _read_insert()
                if text:
                    mt = _read_max_turns()
                    self.stages.insert(0, Stage(prompt=text, max_turns=mt))
                    total += 1
                    self.save()
                    print(f"Inserted. {len(self.stages)} step(s) remaining.")
            # "c" falls through


def _safe_slug(s: str, maxlen: int = 24) -> str:
    """Lowercase, replace non-alphanumeric with hyphens, trim to maxlen."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:maxlen]


def _menu() -> str:
    while True:
        sys.stdout.write("\n[c] continue  [i] insert  [a] abort\n> ")
        sys.stdout.flush()
        choice = sys.stdin.readline().strip().lower()
        if choice in ("c", "i", "a"):
            return choice
        print("  enter c, i, or a")


def _read_insert() -> str:
    print("Prompt to insert (blank line to finish):")
    lines: list[str] = []
    while True:
        sys.stdout.write("  > ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line or line == "\n":
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _read_max_turns(default: int = DEFAULT_MAX_TURNS) -> int:
    sys.stdout.write(f"  max_turns for this step [{default}]: ")
    sys.stdout.flush()
    line = sys.stdin.readline().strip()
    try:
        return int(line) if line else default
    except ValueError:
        return default
