"""The ``eval_tree_model`` tool: the agent's in-turn self-evaluation.

Instead of writing ``get_tree_model.py`` blind and waiting a whole iteration to
see how it did, the agent calls this tool with its full candidate source and
*immediately* gets back, per test case and per port: the rtlsim TARGET vector,
its own vector, and the delta. This is the single biggest change to how the
model experiences the task -- it turns one blind guess per iteration into a
tight fit against a visible target.

The backend (local / docker) is whatever the bound :class:`Oracle` was
configured with; the agent does not know or care which.
"""

from __future__ import annotations

import ast
from pathlib import Path

CANDIDATE_FILENAME = "get_tree_model.py"

EVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "eval_tree_model",
        "description": (
            "Evaluate a candidate get_tree_model against the rtlsim reference and "
            "return, per test case and per port (input/output), the target vector, "
            "your vector, and their delta. Call this as often as you like -- it is "
            "how you see whether a tree is correct. Pass the COMPLETE file contents "
            "(a single top-level `def get_tree_model(self):` returning a "
            "Characteristic_Node). It is saved to the workspace as get_tree_model.py "
            "and becomes your current best candidate. The feedback reports a `score` = "
            "total error summed over every port of every case; LOWER is better and "
            "**0 is the goal** -- score 0 with no fails means every port matches the "
            "reference exactly and the node is perfectly solved. Each call also reports "
            "whether your score improved or worsened versus your previous attempt. Keep "
            "going until every port reports 'exact' (zero delta) and the score is 0."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Full contents of get_tree_model.py (no markdown fences, no imports).",
                }
            },
            "required": ["source"],
        },
    },
}


def make_eval_handler(oracle, *, on_result=None):
    """Return a tool handler closure bound to ``oracle``.

    ``on_result(source, result)`` is an optional callback so the loop can record
    every candidate the agent tries (for the archive / best-tracking).

    The closure remembers this worker's previous score so each call can tell the
    agent whether its latest change improved or worsened the result.
    """
    prev = {"score": None}

    def handler(args: dict, workspace: Path) -> str:
        source = args.get("source", "")
        if not source.strip():
            return "error: `source` was empty. Pass the full get_tree_model.py contents."
        # cheap structural gate before any (possibly expensive) evaluation
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"SyntaxError in your candidate: {e}. Fix it and call eval_tree_model again."
        if not any(
            isinstance(n, ast.FunctionDef) and n.name == "get_tree_model" for n in tree.body
        ):
            return "error: no top-level `def get_tree_model(self):` found. It must be a single top-level function."

        path = workspace / CANDIDATE_FILENAME
        path.write_text(source)

        try:
            result = oracle.evaluate(source, str(path))
        except Exception as e:  # never let a tool error kill the agent loop
            return f"evaluation error ({type(e).__name__}): {e}"

        if on_result is not None:
            try:
                on_result(source, result)
            except Exception:
                pass

        feedback = oracle.format_feedback(result, prev_score=prev["score"])
        prev["score"] = result.score
        if result.solved:
            feedback += (
                "\n\nAll ports match the reference EXACTLY on every active case. "
                "This candidate is saved; the loop will now confirm it with a real "
                "docker run. If confirmed, you are done."
            )
        return feedback

    return handler
