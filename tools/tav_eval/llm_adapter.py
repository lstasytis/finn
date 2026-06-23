"""LLM integration point for the evolve loop.

This is the ONE file you edit to hook in your local model
(tools/Bespoke-Base-Retreat-26). The evolve loop (evolve.py) only ever calls
``propose_candidate(ctx)`` and expects back the source of a new
``get_tree_model`` function. Everything else -- scoring, feedback, keeping the
best candidate -- is handled for you.

Two ready-made ways to wire it in (pick one):

  A) CLI command (default, most portable):
     Point the loop at a shell command with ``--llm-cmd`` or the ``TAV_LLM_CMD``
     environment variable. The command receives the prompt on **stdin** and must
     print the candidate ``get_tree_model`` (python) to **stdout**. e.g.

         export TAV_LLM_CMD='python tools/Bespoke-Base-Retreat-26/generate.py'

  B) Direct python call:
     If Bespoke exposes an importable function, replace the body of
     ``_call_llm`` below with your call, e.g.

         from bespoke import generate
         return generate(prompt)

The returned text may include markdown fences or surrounding prose; it is
sanitized down to a parseable ``get_tree_model`` definition automatically.
"""

import ast
import os
import re
import subprocess
import textwrap


# ---------------------------------------------------------------------------
# >>> EDIT HERE to call Bespoke-Base-Retreat-26 <<<
# ---------------------------------------------------------------------------
def _call_llm(prompt, ctx):
    """Send ``prompt`` to the model and return its raw text response.

    Default behaviour: run the command in ctx['llm_cmd'] (or $TAV_LLM_CMD),
    feeding the prompt on stdin and capturing stdout. Replace the body with a
    direct python/HTTP call if you prefer (option B above)."""
    cmd = ctx.get("llm_cmd") or os.environ.get("TAV_LLM_CMD")
    if not cmd:
        raise RuntimeError(
            "No LLM configured. Set --llm-cmd / $TAV_LLM_CMD to a command that "
            "reads a prompt on stdin and writes a get_tree_model on stdout, or "
            "edit _call_llm() in llm_adapter.py to call Bespoke directly."
        )
    # cmd is a shell string so you can include args/pipes freely
    proc = subprocess.run(
        cmd,
        shell=True,
        input=prompt,
        capture_output=True,
        text=True,
        cwd=ctx.get("repo_root"),
        timeout=ctx.get("llm_timeout", 600),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"LLM command failed (exit {proc.returncode}).\nstderr:\n{proc.stderr[-2000:]}"
        )
    return proc.stdout


# ---------------------------------------------------------------------------
# public entry point used by evolve.py
# ---------------------------------------------------------------------------
def propose_candidate(ctx):
    """Return the source of a new candidate get_tree_model (a python function
    definition) given the evolve context ``ctx``."""
    prompt = build_prompt(ctx)
    raw = _call_llm(prompt, ctx)
    return sanitize_candidate(raw)


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def build_prompt(ctx):
    """Build the optimization prompt from the loop context."""
    fb = ctx.get("feedback", "")
    history = ctx.get("history", [])
    hist_lines = "\n".join(
        f"  iter {h['iteration']}: score={h['score']} "
        f"(pass={h['n_pass']} fail={h['n_fail']} error={h['n_error']})"
        for h in history[-8:]
    )
    return textwrap.dedent(
        f"""\
        You are optimizing the `get_tree_model` method of the FINN hardware node
        "{ctx['node']}". This method returns a tree of `Characteristic_Node`
        objects describing how the node emits/consumes tokens. The tree is used
        to analytically predict a "token access vector" (TAV) that must match the
        ground-truth RTL-simulation TAV as closely as possible (smaller deltas
        are better; zero delta on every test case is the goal).

        Rules for your answer:
        - Output ONLY a single Python function `def get_tree_model(self):` .
        - Do not include imports or prose. You may use any symbol already
          available in the node's module (e.g. `Characteristic_Node`, `np`,
          `self.get_nodeattr(...)`).
        - Keep the same return contract (return the top-level Characteristic_Node).

        Current best `get_tree_model` (score {ctx.get('best_score')}):
        ```python
        {ctx['current_source'].strip()}
        ```

        Most recent evaluation feedback (per test case, delta = analytical - rtlsim):
        {fb if fb else "  (none yet)"}

        Score history (lower is better):
        {hist_lines if hist_lines else "  (none yet)"}

        Propose an improved `get_tree_model` that reduces the TAV deltas.
        """
    )


# ---------------------------------------------------------------------------
# response sanitization
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def sanitize_candidate(text):
    """Extract a parseable `get_tree_model` definition from a raw LLM response.

    Strips markdown fences and surrounding prose. Raises ValueError if no valid
    get_tree_model function can be recovered."""
    # prefer fenced code if present
    m = _FENCE.search(text)
    body = m.group(1) if m else text

    # fast path: already a parseable module containing get_tree_model
    if _has_get_tree_model(body):
        return body.rstrip() + "\n"

    # otherwise, slice from the first `def get_tree_model` to the next
    # top-level (column-0) def/class, then dedent
    lines = body.splitlines()
    start = None
    indent = ""
    for i, ln in enumerate(lines):
        m2 = re.match(r"^(\s*)def\s+get_tree_model\s*\(", ln)
        if m2:
            start = i
            indent = m2.group(1)
            break
    if start is None:
        raise ValueError("LLM response contained no `def get_tree_model(`")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.startswith(indent + " ") and not ln.startswith(indent + "\t"):
            # a line at the same or lower indentation that isn't blank ends the def
            if re.match(r"^\s*(def|class)\s", ln) and (len(ln) - len(ln.lstrip())) <= len(indent):
                end = j
                break
    snippet = textwrap.dedent("\n".join(lines[start:end]))
    if not _has_get_tree_model(snippet):
        raise ValueError("Recovered snippet does not parse as get_tree_model")
    return snippet.rstrip() + "\n"


def _has_get_tree_model(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "get_tree_model"
        for n in ast.walk(tree)
    )
