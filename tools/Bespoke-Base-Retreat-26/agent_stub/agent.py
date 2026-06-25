"""A minimal tool-calling agent loop on the plain ``openai`` client.

Works identically against OpenAI cloud and local vLLM servers -- the router
hands us a client already pointed at the right endpoint, and both speak the
same chat-completions + tool-calling API.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_stub.router import make_client
from agent_stub.tools.apply_patch import APPLY_PATCH_TOOL_SCHEMA, apply_patch
from agent_stub.tools.bash import BASH_TOOL_SCHEMA, run_bash
from agent_stub.tools.run import RUN_TOOL_SCHEMA, run_program

SYSTEM_PROMPT = (
    "You are a coding agent working inside a sandboxed workspace directory. "
    "Use `apply_patch` to create and edit files, `run` to execute a program you "
    "produced (e.g. a Python script), and `bash` for any other shell work. All "
    "file paths are relative to the workspace, and everything you run may only "
    "write inside it. Work step by step. When the task is complete, reply with a "
    "short final message and no tool call."
)

# Domain-expert system prompt for the TAV tree-modeling task. The highest-value
# slot in the context carries the mental model and the iron rule (validate
# against the measured vector), so the per-iteration task prompt can focus on
# the specific node + feedback. Used by examples/tav_tree_model_loop.py.
TAV_SYSTEM_PROMPT = (
    "You are a world-class FPGA dataflow engineer reverse-engineering the "
    "cycle-by-cycle I/O behavior of FINN hardware nodes. Your job: write a Python "
    "function `get_tree_model(self)` that builds a tree of Characteristic_Node "
    "objects whose traversal reproduces a node's Token Access Vector (TAV) EXACTLY.\n\n"
    "Mental model:\n"
    "- A TAV is a per-clock-cycle CUMULATIVE running count: element i is the total "
    "number of tokens read (input port) or written (output port) by cycle i. It is "
    "monotonic non-decreasing. A '1-step' between consecutive elements means a token "
    "moved that cycle; a flat stretch means idle/back-pressure.\n"
    "- A Characteristic_Node is (name, sub_phases, leaf). For a LEAF, sub_phases is a "
    "list of (repeat_count, [read_flag, write_flag]) -- each repeat emits one cycle, "
    "adding read_flag to the input counter and write_flag to the output counter. For a "
    "NON-leaf, sub_phases is a list of (repeat_count, child_node) -- the child's whole "
    "pattern is emitted repeat_count times. So the tree is a compressed, hierarchical "
    "description of the repeating cycle pattern: root = coarse phases (e.g. batch, "
    "rows), leaves = the fine read/write micro-pattern.\n"
    "- Repeat counts come from the node's compile-time attributes via "
    "self.get_nodeattr(...). RTL and HLS backends usually need DIFFERENT trees; branch "
    "on `'_rtl' in self.__class__.__name__`.\n"
    "- A node behaves DIFFERENTLY across parameter regimes (e.g. parallel_window on/off, "
    "stride>1, dilation>1, depthwise). One flat tree rarely fits them all. Branch your "
    "function on self.get_nodeattr(...) and build a distinct sub-tree per regime -- get one "
    "regime exact, then ADD a branch for the next regime WITHOUT changing the branch that "
    "already works. When a change makes already-passing cases regress, you almost certainly "
    "edited a shared path that should have been a separate branch.\n\n"
    "How to work (this matters):\n"
    "1. FIRST call eval_tree_model with the baseline to SEE the target vectors.\n"
    "2. Read the TARGET vector's structure: its period, the run-lengths of 1-steps and "
    "flats, where bursts and idle gaps fall. A quick glance at the reference source for "
    "orientation is fine, but DO NOT sink many turns into reading RTL/HLS or writing "
    "analysis scripts -- the target vectors are the ground truth, and the fast path to "
    "the answer is rapid hypothesize -> eval -> adjust, not full reverse-engineering.\n"
    "3. Encode your hypothesis as a tree and call eval_tree_model again. The tool shows "
    "you TARGET vs YOURS vs DELTA per port. Drive every delta to zero.\n"
    "4. INPUT PORT FIRST. Get EVERY case's input port exact before you work on output -- "
    "input is the simpler port and the foundation the output model builds on. The feedback "
    "tells you which phase you are in and shows full detail only for the port you should be "
    "working on. Once all input ports are exact, switch to output. The overall score (sum "
    "over both ports of all cases) is still what ultimately matters and must reach 0.\n"
    "5. Iterate relentlessly: it is normal and expected to call eval_tree_model MANY times "
    "(dozens). That tight loop -- not source study -- is how you converge. Do not stop "
    "until every port of every case reports 'exact'.\n\n"
    "Hard rules:\n"
    "- Use eval_tree_model as your source of truth, not your imagination, and call it "
    "often. Spending many turns on bash/source-reading instead of eval is the main way to "
    "fail this task. The delta is ground truth about real hardware behavior.\n"
    "- Keep trees small and structural (typically 10-40 nodes). Do NOT write Python "
    "loops that simulate the node cycle-by-cycle; express the pattern as tree structure "
    "with repeat counts derived from get_nodeattr.\n"
    "- The file must contain exactly one top-level `def get_tree_model(self):` and no "
    "import statements (Characteristic_Node, math, np are already in scope). Pass the "
    "FULL file to eval_tree_model each time.\n"
    "- Your function is spliced into the real node, so you MAY call the node's own "
    "sibling methods (e.g. an existing helper tree-builder, or self.get_nodeattr). "
    "self.get_nodeattr(...) is the fast path; calling other node methods still works but "
    "triggers a slower, faithful docker-backed evaluation for that candidate -- so prefer "
    "deriving from get_nodeattr when you can, and reach for helper methods when they "
    "genuinely capture the structure.\n"
    "- You may read reference source under the inputs/ directory with bash; never read "
    "the live FINN checkout."
)

TOOLS = [APPLY_PATCH_TOOL_SCHEMA, RUN_TOOL_SCHEMA, BASH_TOOL_SCHEMA]

# Responses API uses a flatter tool schema (no "function" wrapper).
_TOOLS_RESPONSES = [
    {
        "type": "function",
        "name": s["function"]["name"],
        "description": s["function"]["description"],
        "parameters": s["function"]["parameters"],
    }
    for s in TOOLS
]


def _to_responses_input(messages: list[dict]) -> list:
    """Convert chat-completions message list to Responses API input items.

    The Responses API stores function calls as top-level input items rather than
    inside an assistant message, and tool results use "function_call_output" type
    instead of role="tool". We do this conversion on every call so the pipeline's
    chat-format message history remains the single source of truth for save/resume.
    """
    items: list = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            items.append({"role": "developer", "content": m["content"]})
        elif role == "user":
            items.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m["tool_call_id"],
                "output": m["content"],
            })
    return items


def _trim_tool_history(messages: list[dict], keep_full: int = 2, stub_threshold: int = 1500) -> None:
    """Bound context growth across many tool calls.

    Each ``eval_tree_model`` result can be large (full TARGET/YOURS/DELTA over
    many cases). They accumulate in the message history every call, so even
    individually-bounded feedbacks can blow the model's input limit after a
    dozen calls. Old tool results are for *superseded* candidates -- only the
    latest few matter -- so we replace the content of all but the last
    ``keep_full`` large tool messages with a short stub. The messages themselves
    stay (preserving the assistant tool_call <-> tool result pairing the API
    requires); only their bulky content is dropped. Mutates ``messages`` so the
    trim persists into saved/--memory history too."""
    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    protect = set(tool_idxs[-keep_full:])
    for i in tool_idxs:
        if i in protect:
            continue
        content = messages[i].get("content") or ""
        if len(content) > stub_threshold:
            messages[i]["content"] = (
                "[earlier tool result elided to conserve context -- it was for a superseded "
                "candidate; rely on your most recent evaluation below]"
            )


def _dispatch(name: str, args: dict, workspace: Path, tool_handlers: dict | None = None) -> str:
    try:
        if tool_handlers and name in tool_handlers:
            return tool_handlers[name](args, workspace)
        if name == "bash":
            return run_bash(args["command"], workspace)
        if name == "apply_patch":
            return apply_patch(args["patch"], workspace)
        if name == "run":
            return run_program(args["path"], workspace, args.get("args"))
        return f"error: unknown tool {name!r}"
    except Exception as e:  # surface tool errors to the model, don't crash the loop
        return f"error: {e}"


def _preview(args: dict) -> str:
    s = json.dumps(args)
    return s if len(s) <= 120 else s[:120] + "..."


def _responses_tools(extra_tools):
    base = list(_TOOLS_RESPONSES)
    for s in extra_tools or []:
        base.append(
            {
                "type": "function",
                "name": s["function"]["name"],
                "description": s["function"]["description"],
                "parameters": s["function"]["parameters"],
            }
        )
    return base


def run_agent_with_history(
    messages: list[dict],
    model: str,
    workspace: Path,
    *,
    max_turns: int = 30,
    verbose: bool = True,
    extra_tools: list | None = None,
    tool_handlers: dict | None = None,
) -> tuple[str, list[dict]]:
    """Run the agent loop from an existing message history.

    The caller is responsible for appending the user turn before calling.
    Mutates and returns the same messages list so the caller can persist
    the full conversation across multiple steps.

    ``extra_tools`` are extra chat-completions tool schemas to expose;
    ``tool_handlers`` maps tool name -> ``fn(args, workspace) -> str`` and is
    consulted before the built-in tools, so callers can inject e.g.
    ``eval_tree_model`` without touching this module.
    """
    client, m = make_client(model)
    workspace.mkdir(parents=True, exist_ok=True)

    chat_tools = TOOLS + list(extra_tools or [])
    resp_tools = _responses_tools(extra_tools)
    # sampling/reasoning knobs sourced from the Model (see models.py)
    resp_kwargs = {}
    if getattr(m, "reasoning_effort", None):
        resp_kwargs["reasoning"] = {"effort": m.reasoning_effort}
    chat_kwargs = {"temperature": getattr(m, "temperature", 1.0)}
    if getattr(m, "reasoning_effort", None):
        chat_kwargs["reasoning_effort"] = m.reasoning_effort

    for _ in range(max_turns):
        _trim_tool_history(messages)
        if m.use_responses:
            resp = client.responses.create(
                model=m.name,
                input=_to_responses_input(messages),
                tools=resp_tools,
                tool_choice="auto",
                **resp_kwargs,
            )
            calls = [o for o in resp.output if o.type == "function_call"]
            final_text = "".join(
                b.text
                for o in resp.output if o.type == "message"
                for b in o.content if hasattr(b, "text")
            )
            assistant: dict = {"role": "assistant", "content": final_text}
            if calls:
                assistant["tool_calls"] = [
                    {
                        "id": c.call_id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in calls
                ]
            messages.append(assistant)

            if not calls:
                if verbose and final_text:
                    print(f"\n=== final ===\n{final_text}")
                return final_text, messages

            for c in calls:
                try:
                    args = json.loads(c.arguments or "{}")
                except json.JSONDecodeError:
                    result = "error: tool arguments were not valid JSON"
                else:
                    if verbose:
                        print(f"\n-> {c.name}({_preview(args)})")
                    result = _dispatch(c.name, args, workspace, tool_handlers)
                    if verbose:
                        print("\n".join("   " + l for l in result.splitlines()[:20]))
                messages.append({"role": "tool", "tool_call_id": c.call_id, "content": result})

        else:
            resp = client.chat.completions.create(
                model=m.name,
                messages=messages,
                tools=chat_tools,
                tool_choice="auto",
                **chat_kwargs,
            )
            msg = resp.choices[0].message

            # Re-append assistant turn manually so provider extras (reasoning_content
            # etc.) don't leak into the next request.
            assistant = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant)

            if not msg.tool_calls:
                if verbose and msg.content:
                    print(f"\n=== final ===\n{msg.content}")
                return msg.content or "", messages

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    result = "error: tool arguments were not valid JSON"
                else:
                    if verbose:
                        print(f"\n-> {name}({_preview(args)})")
                    result = _dispatch(name, args, workspace, tool_handlers)
                    if verbose:
                        print("\n".join("   " + l for l in result.splitlines()[:20]))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "[stopped: max turns reached]", messages


def run_agent(
    task: str,
    model: str,
    workspace: Path,
    *,
    max_turns: int = 30,
    verbose: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
    extra_tools: list | None = None,
    tool_handlers: dict | None = None,
    return_messages: bool = False,
):
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    result, msgs = run_agent_with_history(
        messages, model, workspace, max_turns=max_turns, verbose=verbose,
        extra_tools=extra_tools, tool_handlers=tool_handlers,
    )
    if return_messages:
        return result, msgs
    return result
