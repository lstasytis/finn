# Retreat Agent Stub

A sandboxed LLM coding agent. Talks to OpenAI **and** our local vLLM boxes
(dgx01/dgx02) through one code path. Three tools: `apply_patch`, `run`, `bash`.
Read it, fork it.

OpenAI and vLLM speak the same API, so routing is just picking a `base_url`.

## Setup

```bash
uv sync
cp .env.example .env    # only needed for gpt-* models (OPENAI_API_KEY)
```

Linux only (needs Landlock). Local models need the vLLM server running —
see `docs/vllm_server_setup.md` in the parent repo.

## Single-shot task

```bash
python main.py "write game.py, an ascii number-guessing game, then run it"
python main.py --model gpt-5.1 "fix the bug in app.py"

echo "write fireworks.py and run it" | python main.py
```

Default model: `unsloth/gemma-4-31B-it` (dgx01). Default workspace: `./workspace`.

## Pipeline — multi-step with persistent context

Run a sequence of stages. The full conversation carries forward across every
step. State is saved to `/mnt/labstore/Retreat-logs/` after each stage; the log
stores the complete message chain so `--resume` picks up exactly where you left
off — same file, same context.

**Recommended: define stages in a prompts directory**

```bash
python pipeline.py --prompts-dir prompts/calc_challenge
python pipeline.py --prompts-dir prompts/btree --auto-continue
```

Each pipeline is a folder of numbered `.toml` files:

```
prompts/
  btree/
    01_implement.toml
    02_traverse_query.toml
```

```toml
# 01_implement.toml
name = "implement"
max_turns = 20

prompt = """
Write btree.py implementing a B-tree...
"""
```

**Inline prompts**

```bash
python pipeline.py --prompts "write primes.py" "now add unit tests" "document it"
python pipeline.py --prompts-file plan.txt --auto-continue
```

**Resume an interrupted run**

```bash
python pipeline.py --resume /mnt/labstore/Retreat-logs/20260619_101035_teckmann_btree_gemma-4-31b-it.json
```

Logs are named `{timestamp}_{user}_{pipeline}_{model}.json` so the shared folder
stays navigable. Use `--name` to override the pipeline label.

**Between stages** (when not using `--auto-continue`):

```
[c] continue  [i] insert  [a] abort
```

`i` prompts for a new stage that runs next, then continues with the original
queue. The inserted stage is saved to the log immediately.

**All options**

```
--prompts-dir DIR       load stages from a directory of .toml files
--prompts PROMPT ...    define stages inline
--prompts-file FILE     load from text file (blank lines separate stages)
--resume LOG            continue from a saved log file
--extend PROMPT ...     with --resume: append extra stages
--model MODEL           default: unsloth/gemma-4-31B-it
--workspace DIR         default: ./workspace
--log-dir DIR           default: /mnt/labstore/Retreat-logs
--name LABEL            label embedded in the log filename
--auto-continue         skip the interactive menu
--max-turns N           default max turns per stage (overridden per .toml)
```

## Iteration loop — generate → test → fix → repeat

The agent writes a solution, an external test harness checks it, and failures
feed back as the next prompt. Loops until all tests pass or `--max-iterations`.

```bash
python examples/iteration_loop.py
python examples/iteration_loop.py --model unsloth/DeepSeek-V4-Flash --max-iterations 3
```

The default task is an arithmetic expression parser (no `eval()` allowed) with
10 test cases covering precedence, right-associative `**`, and unary minus.
Swap `TASK` and `check_output()` in the file for your own problem.

## Structured outputs

```bash
python examples/structured_output.py
```

Pass a Pydantic model, get a typed object back — no string parsing:

```python
from agent_stub.structured import parse_structured

result = parse_structured("unsloth/gemma-4-31B-it", messages, MyPydanticModel)
```

Works on cloud and local models identically.

## Models

`api_base = None` → OpenAI cloud (needs `OPENAI_API_KEY`). Otherwise hits that
vLLM URL. Model name must match `--served-model-name` on the server.

| Model | Where |
|---|---|
| `gpt-5.1`, `gpt-5.1-codex`, `gpt-5.2-codex` | OpenAI cloud |
| `unsloth/MiniMax-M3`, `unsloth/DeepSeek-V4-Flash` | dgx02 `:13505` |
| `unsloth/gemma-4-31B-it` | dgx01 `:13507` |

Add a model: one line in `agent_stub/models.py`.

## Sandbox

Landlock blocks all writes outside the workspace. `apply_patch`/`run` are
workspace-only; `bash` also gets `/tmp`. Don't widen it — if you need more room,
add a tool instead. The relevant code is `sandbox.py`, the `_safe_path` checks
in `run.py` and `apply_patch.py`, and `writable_roots` in `bash.py` and `run.py`.

## Layout

```
main.py                     single-shot CLI
pipeline.py                 multi-step pipeline CLI
prompts/
  calc_challenge/           example: expression parser + adversarial tester
  btree/                    example: B-tree implementation + traversal/query
agent_stub/
  models.py                 model registry
  router.py                 api_base → OpenAI client
  agent.py                  tool-calling loop + system prompt
  pipeline.py               Pipeline class (Stage, save/load/run)
  structured.py             pydantic structured outputs
  sandbox.py                Landlock sandbox (don't touch)
  tools/
    apply_patch.py          V4A patch → file writes (workspace only)
    run.py                  run a program (workspace only)
    bash.py                 shell (workspace + /tmp)
examples/
  build_fun_program.py      one-shot demo
  structured_output.py      pydantic demo
  iteration_loop.py         generate → test → fix loop
```

## Adding your own pipeline

Drop a folder under `prompts/` with numbered `.toml` files and run it:

```bash
mkdir prompts/my_task
# write 01_step_one.toml, 02_step_two.toml ...
python pipeline.py --prompts-dir prompts/my_task
```

Each `.toml` needs at minimum:

```toml
prompt = """
Your task here.
"""
```

Optional: `name = "label"`, `max_turns = 20`.
