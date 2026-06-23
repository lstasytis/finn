"""Structured outputs with pydantic.

Both OpenAI and vLLM can constrain a model's reply to a JSON schema. The
OpenAI SDK's ``chat.completions.parse`` helper turns a pydantic model into that
schema and parses the response straight back into a typed object. Because the
router hands us an ordinary OpenAI client, the *same* call works against cloud
and local endpoints -- only the base URL differs.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from agent_stub.router import make_client

T = TypeVar("T", bound=BaseModel)


def parse_structured(
    model: str,
    messages: list[dict],
    response_model: type[T],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> T:
    """Call ``model`` and parse its reply into ``response_model`` (a pydantic class)."""
    client, m = make_client(model)
    completion = client.chat.completions.parse(
        model=m.name,
        messages=messages,
        response_format=response_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise RuntimeError(
            "Model did not return a parseable structured response "
            f"(refusal={message.refusal!r})."
        )
    return message.parsed
