"""
RELIABILITY MIDDLEWARE (part of Orchestration)

Guards every agent needs, all built into LangChain 1.x:

  ModelCallLimitMiddleware  - hard cap on model calls per run. Stops runaway loops.
  ModelFallbackMiddleware   - when the primary model raises, try the fallback from .env.
  ModelRetryMiddleware      - retries transient model errors (429, 5xx) with backoff.
  ToolRetryMiddleware       - retries a tool that raised; after the last retry, returns the
                              error as a ToolMessage so the model can recover or explain.

ORDER MATTERS. LangChain composes wrappers first = outermost. So:
  [Fallback, Retry]  = retry the primary N times, THEN fall back      (what we want)
  [Retry, Fallback]  = every retry attempt also tries the fallback    (not what we want)
And retry must re-raise (`on_failure="error"`) for the outer fallback to see the failure.
Verified against langchain 1.4 `_chain_model_call_handlers` / `_chain_tool_call_wrappers`.
"""

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

MAX_MODEL_CALLS_PER_RUN = 25


def _tool_error_text(error: Exception) -> str:
    """What the model reads when a tool keeps failing."""
    return f"error: tool failed after retries: {type(error).__name__}: {error}"


def build_reliability_middleware(fallback_model: BaseChatModel | None = None) -> list:
    middleware = [
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS_PER_RUN, exit_behavior="end"),
    ]
    if fallback_model is not None:
        # Outer: fallback. Inner: retry primary, re-raise when exhausted so fallback runs.
        middleware += [
            ModelFallbackMiddleware(fallback_model),
            ModelRetryMiddleware(max_retries=2, initial_delay=1.0, on_failure="error"),
        ]
    else:
        # No fallback: after retries, hand the model an error message instead of crashing.
        middleware.append(ModelRetryMiddleware(max_retries=2, initial_delay=1.0))
    middleware.append(
        ToolRetryMiddleware(max_retries=1, initial_delay=0.5, on_failure=_tool_error_text)
    )
    return middleware
