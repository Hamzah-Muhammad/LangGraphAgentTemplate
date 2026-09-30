# agent/model/model.py  |  BLOCK 1 MODEL
"""
BLOCK 1: MODEL

The only block that thinks. Everything else is plumbing around it.

Provider is chosen by environment, not code. Any OpenAI-compatible host works:
Groq, NVIDIA NIM, OpenRouter, Together, a local vLLM/Ollama server.
Swap host = change OPENAI_BASE_URL and MODEL_NAME in .env.
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def _model_from_env(prefix: str, required: bool) -> BaseChatModel | None:
    api_key = os.getenv(f"{prefix}API_KEY")
    base_url = os.getenv(f"{prefix}BASE_URL")
    model_name = os.getenv(f"{prefix}MODEL_NAME")
    missing = [
        k
        for k, v in {
            f"{prefix}API_KEY": api_key,
            f"{prefix}BASE_URL": base_url,
            f"{prefix}MODEL_NAME": model_name,
        }.items()
        if not v
    ]
    if missing:
        if required:
            raise RuntimeError(
                f"Missing env vars: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
        return None

    # model_provider="openai" means "speak the OpenAI wire format", not "use OpenAI".
    return init_chat_model(
        model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        stream_usage=True,  # token counts arrive on streamed messages too (agent/shared/usage.py)
    )


def build_model() -> BaseChatModel:
    """Primary model. Raises with a clear message if .env is incomplete."""
    return _model_from_env("OPENAI_", required=True)


def build_fallback_model() -> BaseChatModel | None:
    """Optional second model used when the primary keeps failing. None if not configured."""
    return _model_from_env("FALLBACK_", required=False)


def build_grader_model() -> BaseChatModel | None:
    """Optional model for the evaluator's grader. A different model grades more honestly
    than the one that wrote the draft. None = reuse the primary with a grader prompt."""
    return _model_from_env("GRADER_", required=False)
