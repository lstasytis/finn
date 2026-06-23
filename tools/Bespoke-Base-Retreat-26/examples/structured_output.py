"""Example: a structured "code roast" review parsed into nested pydantic models.

Shows off the parts of structured outputs you actually use day to day:
  * an Enum field (Severity)
  * a list of nested sub-models (Issue)
  * a constrained integer (score 1-10)
The model's reply is constrained to this schema, so you get a fully typed
object back -- no brittle string parsing. Same call works on cloud or local.

Run (default local model on dgx02):
    python examples/structured_output.py
    python examples/structured_output.py --model unsloth/DeepSeek-V4-Flash
"""

from __future__ import annotations

import argparse
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_stub.models import DEFAULT_MODEL
from agent_stub.structured import parse_structured


class Severity(str, Enum):
    nit = "nit"
    minor = "minor"
    major = "major"
    critical = "critical"


class Issue(BaseModel):
    location: str = Field(description="Where the issue is, e.g. a function or line.")
    severity: Severity
    problem: str = Field(description="What is wrong.")
    suggested_fix: str = Field(description="A concrete fix.")


class CodeReview(BaseModel):
    summary: str = Field(description="One-sentence overall summary.")
    roast: str = Field(description="A witty, light-hearted one-liner about the code.")
    issues: list[Issue] = Field(description="Individual problems found, worst first.")
    score: int = Field(ge=1, le=10, description="Overall code quality, 1 (yikes) to 10 (chef's kiss).")
    verdict: str = Field(description="Short call: 'ship it', 'needs work', or 'back to the drawing board'.")


SNIPPET = '''
def get_user(users, id):
    for i in range(len(users)):
        if users[i]["id"] == id:
            return users[i]
    return None

def total(items):
    t = 0
    for i in items:
        t = t + i["price"] * i["qty"]
    return t

password = "hunter2"
'''

SEVERITY_ICON = {
    Severity.nit: ".",
    Severity.minor: "-",
    Severity.major: "!",
    Severity.critical: "X",
}


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Structured 'code roast' demo.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    review = parse_structured(
        args.model,
        [
            {
                "role": "system",
                "content": (
                    "You are a witty but genuinely helpful senior engineer doing a "
                    "code review. Be constructive and specific, and have a little fun."
                ),
            },
            {"role": "user", "content": f"Review this Python:\n```python\n{SNIPPET}\n```"},
        ],
        CodeReview,
    )

    # `review` is a fully typed CodeReview instance.
    bar = "#" * review.score + "." * (10 - review.score)
    print(f"\n  {review.summary}")
    print(f"  roast: {review.roast}")
    print(f"  score: [{bar}] {review.score}/10  ->  {review.verdict}\n")
    print(f"  issues ({len(review.issues)}):")
    for issue in review.issues:
        print(f"    {SEVERITY_ICON[issue.severity]} [{issue.severity.value}] {issue.location}")
        print(f"        problem: {issue.problem}")
        print(f"        fix:     {issue.suggested_fix}")

    print("\n  --- raw typed object (model_dump_json) ---")
    print(review.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
