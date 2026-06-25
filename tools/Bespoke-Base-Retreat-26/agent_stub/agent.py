"""A minimal tool-calling agent loop on the plain ``openai`` client.

Works identically against OpenAI cloud and local vLLM servers -- the router
hands us a client already pointed at the right endpoint, and both speak the
same chat-completions + tool-calling API.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from agent_stub.router import make_client
from agent_stub.tools.apply_patch import APPLY_PATCH_TOOL_SCHEMA, apply_patch
from agent_stub.tools.bash import BASH_TOOL_SCHEMA, run_bash
from agent_stub.tools.run import RUN_TOOL_SCHEMA, run_program

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a coding agent working inside a sandboxed workspace directory.
    Use `apply_patch` to create and edit files, `run` to execute a program you
    produced (e.g. a Python script), and `bash` for any other shell work. All
    file paths are relative to the workspace, and everything you run may only
    write inside it. Only read files inside the location(s) your task message
    names as available to you -- never read, list, or grep anything else on
    the filesystem, even if it's technically reachable; if something you need
    isn't there, say so instead of going to look for it elsewhere. Work step
    by step. When the task is complete, reply with a short final message and
    no tool call.

    You are an FPGA expert in Vitis HLS and SystemVerilog, working with
    characteristic tree models of ML operators. A tree model is one Python
    function, get_tree_model(self), returning a Characteristic_Node: a node
    holding a list of child states plus how many times each repeats (its
    edges), where each leaf state flags whether it reads, writes, both, or
    neither. Traversing the tree end to end produces a token access vector
    (TAV) -- one +1 per read/write flag hit, cycle by cycle -- which is
    compared against an rtl-simulated ground truth. The goal in every task you
    are given is a tree whose TAV is identical to rtlsim's across every test
    case; the tree only needs to capture input/output channel activity, not
    full datapath behavior, and rarely needs more than 10-40 states.

    Approach: extract the operator's compile-time parameters (e.g. via
    self.get_nodeattr(...)) -- these determine which states exist and their
    repeat counts, and should be encoded into the tree's edges. The most
    common mistake is misjudging when a read and a write overlap in the same
    cycle. Build incrementally: first correct volume (total tokens read/
    written matches rtlsim), then correct length (fuse/split phases or add
    idle states so cycle count matches), then exact equality (fusing reads/
    writes and partial states correctly, cycle by cycle). Evaluation feedback
    reports each test case's input and output ports as separate pass/fail
    results -- you don't need both sides right at once; it's usually easier
    to get one port passing across all cases first, then build on that
    working tree to get the other port correct, rather than fixing both at
    once.

    Work in this order:
    1. Start from the existing tree and see how it scores.
    2. Look at the per-case deltas and adjust edges/states to shrink them.
    3. Once one test case passes, adjust the tree to pass another -- this
       will likely break the first one again.
    4. Find a single tree structure that covers both cases at once.
    5. Repeat, adding one more case at a time once the previous ones hold.

    DO NOT ATTEMPT TO SIMULATE THE OPERATOR'S RTL/HLS BEHAVIOR YOURSELF, by
    hand-reasoning or in code (e.g. with for-loops). The evaluation feedback
    you are given already reflects the ground truth of how the real hardware
    translates to reads and writes -- treat it as the only authority on
    behavior, and reason about tree structure, not hardware semantics.
""")

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


def _dispatch(name: str, args: dict, workspace: Path) -> str:
    try:
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


def run_agent_with_history(
    messages: list[dict],
    model: str,
    workspace: Path,
    *,
    max_turns: int = 30,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """Run the agent loop from an existing message history.

    The caller is responsible for appending the user turn before calling.
    Mutates and returns the same messages list so the caller can persist
    the full conversation across multiple steps.
    """
    client, m = make_client(model)
    workspace.mkdir(parents=True, exist_ok=True)

    for _ in range(max_turns):
        if m.use_responses:
            resp = client.responses.create(
                model=m.name,
                input=_to_responses_input(messages),
                tools=_TOOLS_RESPONSES,
                tool_choice="auto",
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
                    result = _dispatch(c.name, args, workspace)
                    if verbose:
                        print("\n".join("   " + l for l in result.splitlines()[:20]))
                messages.append({"role": "tool", "tool_call_id": c.call_id, "content": result})

        else:
            resp = client.chat.completions.create(
                model=m.name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=1.0,
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
                    result = _dispatch(name, args, workspace)
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
) -> str:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    result, _ = run_agent_with_history(
        messages, model, workspace, max_turns=max_turns, verbose=verbose
    )
    return result
