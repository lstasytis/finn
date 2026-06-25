# Bespoke-Base-Retreat-26

A sandboxed LLM coding-agent stub (`agent_stub`) plus a set of example loops that
drive it. Talks to OpenAI cloud and local vLLM boxes (dgx01/dgx02) through one
OpenAI-compatible code path; the endpoint is chosen by `api_base` in
`agent_stub/models.py`. Three agent tools only: `apply_patch`, `run`, `bash`,
all confined to a workspace by a Landlock sandbox (`agent_stub/sandbox.py`).

The most important file here is **`examples/tav_tree_model_loop.py`** — the
target we want to optimize. Everything below focuses on it.

## Repo map (only what matters for the loop)

```
examples/tav_tree_model_loop.py   THE FILE — the generate→eval→reprompt loop
agent_stub/
  agent.py        run_agent() — the tool-calling loop each iteration drives
  models.py       model registry (DEFAULT_MODEL = unsloth/gemma-4-31B-it)
  router.py       api_base -> OpenAI client
  tools/          apply_patch / run / bash (sandboxed, workspace-only)
inputs/           CLEANED mirror of FINN source the agent is allowed to read
                  (deps/finn-hlslib, finn-rtllib, src/...), headers stripped
outputs/          run logs: tree-model-run-<node>-<ts>.log (per node),
                  progress_table.txt (shared delta-ratio table), and
                  agents/<node>-iterNNN-popW.log (per-agent transcripts:
                  prompts, tool calls, feedback, final reply)
workspace/        agent's sandbox; get_tree_model.py is written here each iter
../tav_eval/      THE EVALUATOR (sibling dir, added to sys.path, not vendored)
  tav_eval.py     evaluate_tree_model / score_records / docker+pytest plumbing
  examples/<node>_tree_model.py   baseline get_tree_model per node
```

`../tav_eval` lives one level up (`tools/tav_eval`), imported via `sys.path`
insertion at the top of the loop file. It is NOT part of this package.

---

## NEW optimizer architecture (2026-06 rewrite)

The loop was rebuilt to maximize **solve rate** (cost/turns are explicitly not a
concern). Key idea: stop treating this as blind black-box optimization and make
it white-box reverse engineering against a **visible, cached target**.

New/changed files:
```
agent_stub/tav_runtime.py   FAITHFUL pure-Python copy of FINN's analytical TAV
                            derivation (Characteristic_Node + lifted
                            derive_token_access_vectors_using_tree_model, incl.
                            the two-period + micro-buffer-correction logic).
                            Runs on the host, no numpy/finn/docker needed.
agent_stub/oracle.py        Oracle: scores a candidate (local/docker/both),
                            builds cases from plugin records, renders
                            TARGET/YOURS/DELTA feedback, serializes docker evals.
agent_stub/tools/eval_tree.py  eval_tree_model tool: the agent's in-turn self-eval.
agent_stub/progress.py      delta-ratio (|rtlsim-model|/rtlsim) reporting + the
                            shared, flock-guarded progress table written to
                            outputs/progress_table.txt after every iteration of
                            every node (multiple node loops share one table).
agent_stub/agent.py         + TAV_SYSTEM_PROMPT (domain expert); run_agent now
                            takes system_prompt/extra_tools/tool_handlers and
                            passes temperature + reasoning_effort to the API.
agent_stub/models.py        Model gains temperature + reasoning_effort; DEFAULT
                            is now gpt-5.1 @ high effort.
../tav_eval/_tav_eval_plugin.py  now captures, per case: the raw rtlsim_vector
                            (the optimization TARGET), the analytical_vector
                            (for the local-oracle self-test) and node_meta
                            (class_name, onnx_node_name, op_type, full node_attrs)
                            so the host oracle can replay get_tree_model faithfully.
```

**Why the local oracle is faithful, not an approximation.** The dockerized
analytical TAV is NOT a raw tree traversal — `derive_token_access_vectors_using_tree_model`
([hwcustomop.py:329](inputs/src/finn/custom_op/fpgadataflow/hwcustomop.py#L329))
traverses two periods and applies a node-type-specific `apply_micro_buffer_correction`.
`tav_runtime.py` lifts that logic **verbatim**, so given the captured node attrs
it computes byte-identical vectors. `Oracle.selftest()` proves this on the
baseline before any LLM iteration trusts it; and a locally-"solved" candidate is
always **confirmed with a real docker run** before it counts (the rtlsim
reference is cached, so docker is the source of truth either way).

### CLI args
```
node                      FINN node(s) (positional); comma-separate to optimize
                          several in parallel, e.g. `MVAU, FMPadding` (one OS
                          process per node, shared progress table)
--model M                 default gpt-5.1 (@ high reasoning effort); any models.py id
--progress-table PATH     shared progress table (default outputs/progress_table.txt);
                          flock-guarded so concurrent node loops append safely
--parallel N              population: N agents/iteration at spread temperatures (default 4)
--oracle {local,docker,both}  self-eval backend (default local, docker-confirmed)
--curriculum              start on smallest case, widen one at a time
--memory                  carry the best lineage's conversation across iterations
--analyzer                second agent each round for structural advice
--recombine K             feed back up to K archived partial solutions to merge
--no-docker-confirm       trust the local oracle without a docker gate (faster, unverified)
--reasoning-effort {low,medium,high}   override the model's effort
--max-iterations N (20)   --max-turns N (60, per worker per iteration)
--apply-best  --baseline  --src  --test  --extra-tests  --cache-dir  --out  --log
```

### Loop flow
1. **One baseline docker run** captures reference vectors + node metadata →
   `Oracle.load_cases`. (This is the only mandatory docker hit before iterating.)
2. **Self-test** the local oracle against the captured analytical vectors.
3. Each iteration: launch `--parallel` agents (domain system prompt + task),
   each with `eval_tree_model` bound to the oracle; every candidate they try is
   pooled into a thread-safe **Archive**. Global best (lowest score on the active
   case set) is kept; `--recombine` feeds partial solutions back.
4. **Curriculum** widens the active case set as cases are solved.
5. When the best solves all cases locally, **confirm in docker**; on confirm,
   done. If docker disagrees (oracle drift), the real delta is fed back.
6. Restore FINN source/test files; `--apply-best` splices the winner in.

Test the pure-Python core without docker/model:
`python3 agent_stub/tav_runtime.py` (derivation smoke test).

---

## What the ORIGINAL `tav_tree_model_loop.py` did (superseded, kept for context)

It is a closed-loop optimizer that makes an LLM iteratively rewrite a single
FINN function — `get_tree_model()` — for one hardware node until the node's
analytical model matches its RTL-simulation ground truth.

### Domain in one paragraph
FINN models each FPGA dataflow node with a *characteristic tree*
(`Characteristic_Node`, defined in FINN's `src/finn/util/basic.py`). Traversing
the tree emits a **Token Access Vector (TAV)**: per clock cycle, did the node
read an input token and/or write an output token. Leaves are tuples like
`[read_flag, write_flag]`; edges carry repeat counts derived from the node's
compile-time attrs (`self.get_nodeattr(...)`). The goal is a tree whose TAV is
**element-wise identical** to the `rtlsim` reference TAV across every pytest
parametrization. ~10–40 tree nodes suffice for any node; the agent must NOT
simulate the node with loops — it infers the unique repeating states.

### The loop (function: `run_loop`)
1. **Setup.** Resolve node → source path + pytest nodeid via
   `tav_eval.resolve_node`. Load the baseline `get_tree_model` from
   `../tav_eval/examples/<node>_tree_model.py`. Redirect stdout/stderr through
   `_TeeStream` into a timestamped log under `outputs/`.
2. **Baseline pass (iteration 0).** Write the baseline into
   `workspace/get_tree_model.py`, evaluate it, and feed that real delta to the
   analyzer — so iteration 1's prompt starts with genuine feedback, not blind.
   A baseline that already passes short-circuits the whole run.
3. **Iterations 1..max_iterations.** Each iteration:
   - `build_task(...)` renders `TASK_HEADER` (+ `RETRY_SUFFIX` carrying the
     previous candidate and last feedback).
   - `run_agent(task, model, workspace)` — the tree-builder agent writes a new
     `get_tree_model.py` into the workspace (tool calls: apply_patch/run/bash).
   - `check_output(...)` evaluates it (see below), returns
     `(passed, feedback, records, score, pending_case)`.
   - Candidate snapshot saved to `<out_dir>/candidates/iter_NNN.py`; best
     (lowest score) tracked. Stop early when `passed`.
4. **Teardown.** Write `best_get_tree_model.py` + `history.json`, restore the
   node source AND the test file (reverting any analyzer-added cases) via
   `tav_eval.restore_original`. With `--apply-best`, splice the best candidate
   back into the live FINN source.

### `check_output` — the evaluator wiring (the part NOT meant to change)
- Calls `tav_eval.evaluate_tree_model(...)` which: splices the candidate into
  the node's FINN source, runs the characterization pytest **inside the
  `finn_dev_<user>` docker container** via `docker exec`, and returns per-case
  records. rtlsim is served from a **cache** (`$FINN_BUILD_DIR`), so it runs at
  most once per param set; the analytical tree is recomputed every run.
- `score_records` → fitness, **lower is better**, `score==0 & no fail/error ⇒
  solved`. Score is per-port summed absolute delta + length mismatch,
  normalized by reference length × 100. ERROR cases get a 1,000,000 penalty.
- **Two agents per iteration, same model, separate fresh contexts:**
  1. *tree-builder* (`TASK_HEADER`) — writes the candidate.
  2. *analyzer* (`ANALYSIS_TASK`) — read-only review of candidate + this/last
     feedback; emits plain-text structural advice appended to the next
     tree-builder prompt. It may also emit one
     `PROPOSE_TEST_CASE: <param>=<literal>` line to widen coverage by a single
     parametrize value (`_apply_proposed_test_case`), validated/rolled back next
     iteration by `_check_pending_test_case`. `mode`/`impl_style` proposals and
     oversized values are rejected.
- Feedback to the model is **run-length encoded** delta vectors via `_rle` /
  `_fmt_rle`: `(480,-1)` = next 480 cycles each off by −1; target is a single
  `(len,0)` pair per port.

### Inputs / outputs / key constants
- `CANDIDATE_FILENAME = "get_tree_model.py"` — the one file the agent writes.
- `NODE_REFS` — per-node HLS/RTL reference file pointers spliced into prompts.
- `INPUTS_DIR` (`inputs/`) — the ONLY tree the agent may read; prompts forbid it
  from touching the live FINN checkout (`tav_eval.FINN_ROOT`).
- CLI: `python examples/tav_tree_model_loop.py <Node> [--model M]
  [--max-iterations N] [--max-turns N] [--apply-best] [--src ...] [--test ...]`.
  Known nodes: FMPadding, ConvolutionInputGenerator, LabelSelect, Thresholding,
  StreamingDataWidthConverter, MVAU, VVAU, Pool (+ DuplicateStreams baseline).

---

## Where the time goes (read this before optimizing)

The loop is **wall-clock dominated by the agents and the evaluator, not by any
Python here.** Per iteration the cost is, roughly in order:

1. **`run_agent` for the tree-builder** — up to `--max-turns` (default 30) LLM
   round-trips, each a full chat-completions/responses call. Local vLLM or
   OpenAI latency dominates.
2. **`tav_eval.evaluate_tree_model`** — a `docker exec` of `python -m pytest`
   inside the FINN container, once per iteration. Container cold-start
   (`ensure_container`, up to 1800s the first time) and pytest collection/run
   are the heavy parts. rtlsim itself is cached, so repeat param sets are cheap.
3. **`run_agent` for the analyzer** — another full agent run, same `--max-turns`
   budget, every non-passing iteration.

So each iteration is **two full agent runs + one dockerized pytest**. Promising
optimization directions (for later):
- **Parallelism:** iterations are sequential and share one workspace/source
  splice, so they can't trivially overlap; but the analyzer agent could overlap
  the next tree-builder, or multiple nodes/candidates could run concurrently in
  separate containers/workspaces.
- **Agent turn budget:** 30 turns × 2 agents is the main token/latency sink;
  most candidates are a single small file rewrite. Tighter prompts / lower
  max_turns / early-exit when the file stabilizes are levers.
- **Evaluator overhead:** one `docker exec` round-trip per iteration; batching
  candidates or keeping a warm pytest process would cut fixed cost. rtlsim cache
  hits are already the fast path — keep them hits.
- **Feedback size:** RLE keeps deltas compact; watch prompt growth from
  accumulated previous-candidate + previous-feedback + analyzer text.

**Invariant when optimizing:** `check_output` and the `tav_eval` interface are
the evaluation contract and should stay behavior-stable. The prompt text
(`TASK_HEADER`, `RETRY_SUFFIX`, `ANALYSIS_TASK`) is explicitly the hand-tuned
part — changing wording does not require touching the eval side.

---

## Running it

```bash
uv sync
cp .env.example .env          # only for gpt-* models (OPENAI_API_KEY)
python examples/tav_tree_model_loop.py ConvolutionInputGenerator
python examples/tav_tree_model_loop.py FMPadding --model gpt-5.1 --max-iterations 10
python examples/tav_tree_model_loop.py ConvolutionInputGenerator --apply-best
# optimize several nodes in parallel (one process each, shared progress table):
python examples/tav_tree_model_loop.py "MVAU, FMPadding" --model gpt-5.1-codex --max-iterations 10
```

Each node prints an end-of-run summary (wall-clock minutes, iteration count, and
the baseline-vs-final delta ratio `|rtlsim-model|/rtlsim`) and appends a row per
iteration to the shared `outputs/progress_table.txt`:

```
node_name  iteration  max_delta_ratio %  average_delta_ratio %
```

Requirements: Linux (Landlock); a running/Buildable `finn_dev_<user>` docker
container; for local models, the vLLM server up (see parent repo
`docs/vllm_server_setup.md`). Default model is `unsloth/gemma-4-31B-it` (dgx01).

## Conventions

- Agent tools are **workspace-only**; don't widen the sandbox — add a tool
  instead (`agent_stub/sandbox.py`, `_safe_path` in tools).
- The agent reads only `inputs/`; never point it at the live FINN repo.
- Put any new tests under a `tests/` folder and run via `python -m pytest`.
</content>
</invoke>
