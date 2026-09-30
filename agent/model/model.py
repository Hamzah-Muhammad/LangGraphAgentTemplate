# agent/model/model.py  |  BLOCK 1 MODEL
"""
BLOCK 1: MODEL

The only block that thinks. Everything else is plumbing around it.

The provider is chosen in .env, not in code. Three roles, each with its own provider:

    role      provider variable   model variables
    primary   MODEL_PROVIDER      OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL_NAME
                                  or CLAUDE_MODEL when the provider is claude-code
    fallback  FALLBACK_PROVIDER   FALLBACK_API_KEY / FALLBACK_BASE_URL / FALLBACK_MODEL_NAME
    grader    GRADER_PROVIDER     GRADER_API_KEY / GRADER_BASE_URL / GRADER_MODEL_NAME

Providers:
    openai       (default) any OpenAI-compatible host: Groq, NVIDIA NIM, OpenRouter,
                 Together, a local vLLM/Ollama server, OpenAI itself.
    claude-code  Claude through your Claude Pro/Max login, no API key
                 (agent/model/claude_code.py). Only a model name is needed.

So you can draft on Claude and grade on a free Groq model, or the reverse.

HOW TO SWITCH (the README section "Choose your model" has the full steps):
    API-style LLM -> Claude login   set MODEL_PROVIDER=claude-code and CLAUDE_MODEL=sonnet
    Claude login -> API-style LLM   set MODEL_PROVIDER=openai (OPENAI_* lines filled in)
    one run only                    python run.py --provider claude-code --model haiku
    which one is active?            every start prints "[model] primary = ..." (describe_models)

The switch is read once, here, at startup: _provider() picks the branch in
_model_from_env(). Nothing else in the template knows or cares which provider it got.
"""

import importlib.util
import os
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agent.shared.settings import get_settings

PROVIDERS = ("openai", "claude-code")


class ConfigError(RuntimeError):
    """A setting is missing or wrong. The CLI prints the message without a traceback."""

DEFAULT_CLAUDE_MODEL = "sonnet"


def _provider(prefix: str) -> str:
    variable = "MODEL_PROVIDER" if prefix == "OPENAI_" else f"{prefix}PROVIDER"
    provider = (os.getenv(variable) or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise ConfigError(f"{variable}={provider!r} is not supported. Use one of {PROVIDERS}.")
    return provider


def _claude_code_model(prefix: str) -> BaseChatModel:
    if importlib.util.find_spec("claude_agent_sdk") is None:
        raise ConfigError(
            "MODEL_PROVIDER=claude-code needs the Claude Agent SDK. Install it with:\n"
            '    pip install -e ".[claude]"\n'
            "and make sure Claude Code is installed and signed in (run `claude`, then /login)."
        )
    from agent.model.claude_code import ChatClaudeCode  # lazy: the SDK is an optional extra

    variable = "CLAUDE_MODEL" if prefix == "OPENAI_" else f"{prefix}MODEL_NAME"
    return ChatClaudeCode(
        model=os.getenv(variable) or DEFAULT_CLAUDE_MODEL,
        timeout_s=get_settings().request_timeout_s,
    )


def _model_from_env(prefix: str, required: bool) -> BaseChatModel | None:
    if _provider(prefix) == "claude-code":
        return _claude_code_model(prefix)

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
            raise ConfigError(
                f"Missing env vars: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in, "
                "or set MODEL_PROVIDER=claude-code to use your Claude login."
            )
        return None

    # model_provider="openai" means "speak the OpenAI wire format", not "use OpenAI".
    return init_chat_model(
        model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        stream_usage=True,  # token counts arrive on streamed messages too
        # Every request has a deadline. Without one a stalled host hangs the run.
        timeout=get_settings().request_timeout_s,
        max_retries=2,  # client-level retry with backoff for 429 / 5xx / timeouts
    )


ROLES = {"primary": "OPENAI_", "fallback": "FALLBACK_", "grader": "GRADER_"}


def describe_model(prefix: str) -> str | None:
    """One line saying which model a role will use. None if the role is not configured."""
    if _provider(prefix) == "claude-code":
        variable = "CLAUDE_MODEL" if prefix == "OPENAI_" else f"{prefix}MODEL_NAME"
        name = os.getenv(variable) or DEFAULT_CLAUDE_MODEL
        return f"claude-code: {name} (Claude login, no API key)"
    name = os.getenv(f"{prefix}MODEL_NAME")
    if not name:
        return None
    host = urlparse(os.getenv(f"{prefix}BASE_URL") or "").netloc or "no base URL set"
    return f"openai-compatible: {name} @ {host}"


def describe_models() -> list[str]:
    """Banner lines for the CLI: which provider and model each configured role uses."""
    lines = []
    for role, prefix in ROLES.items():
        described = describe_model(prefix)
        if described or role == "primary":
            lines.append(f"[model] {role} = {described or 'NOT CONFIGURED'}")
    return lines


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
