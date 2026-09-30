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


def build_model() -> BaseChatModel:
    """Return the chat model the agent loop will call."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("MODEL_NAME")

    missing = [k for k, v in {
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": base_url,
        "MODEL_NAME": model_name,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env and fill them in."
        )

    # model_provider="openai" means "speak the OpenAI wire format", not "use OpenAI".
    return init_chat_model(
        model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
