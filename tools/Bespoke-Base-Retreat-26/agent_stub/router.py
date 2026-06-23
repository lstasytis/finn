"""LLM router.

Deliberately tiny: because OpenAI *and* our local vLLM servers both speak the
OpenAI API, "routing" is just choosing a ``base_url`` + ``api_key`` for the
``openai`` client based on the model's ``api_base``. One client class, two
destinations.
"""

from __future__ import annotations

import os

from openai import OpenAI

from agent_stub.models import Model, get_model


def make_client(model: str | Model) -> tuple[OpenAI, Model]:
    """Return an ``OpenAI`` client pointed at the right endpoint for ``model``.

    * Cloud model  -> default OpenAI endpoint, needs ``OPENAI_API_KEY``.
    * Local model  -> the model's ``api_base`` (vLLM); key is unchecked, so a
      dummy is fine (override with ``LOCAL_API_KEY`` if your server enforces one).
    """
    m = model if isinstance(model, Model) else get_model(model)

    if m.api_base is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY is not set, required for cloud model {m.name!r}."
            )
        return OpenAI(api_key=api_key), m

    api_key = os.environ.get("LOCAL_API_KEY", "dummy")
    return OpenAI(base_url=m.api_base, api_key=api_key), m
