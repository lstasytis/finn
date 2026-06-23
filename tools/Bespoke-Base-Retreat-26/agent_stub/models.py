"""Model registry for the retreat agent stub.

Each entry describes one model the agent can talk to. The only field the
router really cares about is ``api_base``:

  * ``api_base = None``                       -> OpenAI cloud (uses OPENAI_API_KEY)
  * ``api_base = "http://dgx0X:PORT/v1"``     -> a local vLLM server

vLLM serves an *OpenAI-compatible* API, so the same ``openai`` client library
talks to both cloud and local endpoints. The only thing that changes is the
base URL (and the API key). That is the whole trick behind the router.

To add a model: drop a new entry in ``MODELS``. The ``name`` must be exactly
what the endpoint expects as the ``model`` field (for our vLLM servers that is
the ``--served-model-name``, e.g. ``unsloth/MiniMax-M3``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Local vLLM endpoints. Keep in sync with the running servers.
DGX02 = "http://dgx02:13505/v1"  # Hopper+ (H200): MiniMax-M3, DeepSeek-V4-Flash, ...
DGX01 = "http://10.0.2.51:13507/v1"  # A100: Gemma 4, ...


@dataclass(frozen=True)
class Model:
    name: str  # exact id sent as the `model` field to the endpoint
    api_base: str | None  # None -> OpenAI cloud; otherwise a local vLLM /v1 URL
    context_window: int
    use_responses: bool = False  # True -> POST /v1/responses (Codex models)

    @property
    def is_local(self) -> bool:
        return self.api_base is not None


MODELS: dict[str, Model] = {
    # --- OpenAI cloud (api_base=None -> default OpenAI endpoint) ---
    "gpt-5.1": Model("gpt-5.1", None, 400_000),
    "gpt-5.1-codex": Model("gpt-5.1-codex", None, 400_000, use_responses=True),
    "gpt-5.2-codex": Model("gpt-5.2-codex", None, 400_000, use_responses=True),
    # --- local vLLM on dgx02 (Hopper+) ---
    "unsloth/MiniMax-M3": Model("unsloth/MiniMax-M3", DGX02, 376_832),
    "unsloth/DeepSeek-V4-Flash": Model("unsloth/DeepSeek-V4-Flash", DGX02, 262_144),
    # --- local vLLM on dgx01 ---
    "unsloth/gemma-4-31B-it": Model("unsloth/gemma-4-31B-it", DGX01, 262_144),
}

DEFAULT_MODEL = "unsloth/gemma-4-31B-it"


def get_model(name: str) -> Model:
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise KeyError(f"Unknown model {name!r}. Known models: {known}")
    return MODELS[name]
