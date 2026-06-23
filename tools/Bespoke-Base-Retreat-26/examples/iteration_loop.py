"""Iteration loop: generate → run → evaluate → reprompt.

The agent writes a solution file. After each attempt, check_output() runs a
deterministic test suite against it. On failure, the structured test results
feed back as the next prompt. The loop stops when all tests pass or we hit
--max-iterations.

The default task is full text justification (LeetCode #68). It has enough
edge cases — uneven space distribution, single-word lines, last-line handling
— that the model usually needs 2-3 iterations to get every case right. That
makes it a good stress-test for the feedback loop itself.

Swap out TASK and check_output() for your own problem and success condition.

Run:
    python examples/iteration_loop.py
    python examples/iteration_loop.py --model unsloth/DeepSeek-V4-Flash
    python examples/iteration_loop.py --max-iterations 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from agent_stub.agent import run_agent
from agent_stub.models import DEFAULT_MODEL


TASK = textwrap.dedent("""\
    Write `calc.py` containing a single function:

        evaluate(expr: str) -> float

    that evaluates an arithmetic expression string. Requirements:
    - Operators: + - * / ** with standard mathematical precedence
      (** binds tightest, then * /, then + -)
    - ** is RIGHT-associative: 2 ** 3 ** 2  ==  2 ** (3 ** 2)  ==  512
    - + - * / are LEFT-associative: 10 - 3 - 2  ==  (10 - 3) - 2  ==  5
    - Unary minus is supported: -3, -(4+2), 2**-1
    - Parentheses override precedence in the usual way
    - Whitespace anywhere in the expression is ignored
    - You may assume the expression is well-formed
    - Do NOT use eval(), exec(), compile(), or any parsing library.
      Write the parser yourself.

    Run the file once to confirm there are no syntax errors.
""")

# ── test suite ────────────────────────────────────────────────────────────────

# (expression, expected_float, description)
CASES = [
    ("3 + 4 * 2",          11.0,   "* binds tighter than +"),
    ("(3 + 4) * 2",        14.0,   "parentheses override precedence"),
    ("-3 * -2",             6.0,   "unary minus on both operands"),
    ("2 ** 3 ** 2",        512.0,  "** is right-associative: 2**(3**2)"),
    ("10 - 3 - 2",          5.0,   "- is left-associative: (10-3)-2"),
    ("10 / 4",              2.5,   "float division"),
    ("-(3 + 2)",           -5.0,   "unary minus applied to grouped expr"),
    ("2 ** -1",             0.5,   "negative exponent via unary minus"),
    ("2 * 3 + 4 * 5",      26.0,  "two independent products summed"),
    ("((2+3)*(4-1))**2",  225.0,  "nested parens then exponentiation"),
]

# Injected into a subprocess so the workspace module is imported cleanly.
_TEST_DRIVER = """
import sys, json
sys.path.insert(0, {workspace!r})
try:
    from calc import evaluate
except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e), "failures": []}}))
    sys.exit(0)

cases = {cases!r}
failures = []
for expr, expected, desc in cases:
    try:
        got = evaluate(expr)
    except Exception as e:
        failures.append({{"desc": desc, "expr": expr,
                          "expected": expected, "got": str(e)}})
        continue
    if abs(float(got) - expected) > 1e-9:
        failures.append({{"desc": desc, "expr": expr,
                          "expected": expected, "got": got}})

print(json.dumps({{"ok": not failures, "failures": failures}}))
"""


def check_output(workspace: Path) -> tuple[bool, str]:
    """Return (passed, feedback_for_next_prompt).

    Replace this with your own test logic. The feedback string lands verbatim
    in the next iteration's prompt, so be specific: show expected vs actual.
    """
    script = workspace / "calc.py"
    if not script.exists():
        return False, "calc.py was not created in the workspace."

    code = _TEST_DRIVER.format(workspace=str(workspace), cases=CASES)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if proc.returncode != 0:
        return False, f"Test runner crashed:\n{proc.stderr.strip()}"

    try:
        result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        return False, f"Unexpected output from test runner:\n{proc.stdout.strip()}"

    if "error" in result:
        return False, f"Import error: {result['error']}"

    if result["ok"]:
        return True, ""

    lines = [f"{len(result['failures'])} of {len(CASES)} test cases failed:\n"]
    for f in result["failures"]:
        lines.append(f"  [{f['desc']}]  expr: {f['expr']!r}")
        lines.append(f"    expected: {f['expected']}")
        lines.append(f"    got:      {f['got']}\n")
    return False, "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Iteration loop demo.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--max-iterations", type=int, default=5)
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    print(f"model={args.model}  workspace={workspace}\n")

    feedback = ""
    for i in range(1, args.max_iterations + 1):
        print(f"\n{'='*60}\niteration {i}\n{'='*60}")

        task = (
            TASK if not feedback
            else f"{TASK}\nPrevious attempt failed:\n{feedback}"
        )
        run_agent(task, args.model, workspace)

        passed, feedback = check_output(workspace)
        if passed:
            print(f"\nAll {len(CASES)} test cases passed on iteration {i}.")
            return

        print(f"\nTest feedback:\n{feedback}")

    print(f"\nStopped after {args.max_iterations} iterations.")


if __name__ == "__main__":
    main()
