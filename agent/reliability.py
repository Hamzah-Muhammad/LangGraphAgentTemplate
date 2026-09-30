"""
RELIABILITY MIDDLEWARE (part of Orchestration)

Guards every agent needs, all built into LangChain 1.x:

  ModelCallLimitMiddleware  - hard cap on model calls per run. Stops runaway loops.
  ModelRetryMiddleware      - retries transient model errors (429, 5xx) with backoff.
  ModelFallbackMiddleware   - after retries fail, try the fallback model from .env.
  ToolRetryMiddleware       - retries a tool that raised, with backoff.
  ToolErrorMiddleware       - a tool that still fails returns an error message to the
                              model instead of crashing the run. The model can recover.

Order matters: these run outermost so they wrap the context and approval layers.
"""

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel

MAX_MODEL_CALLS_PER_RUN = 25


def _tool_error_to_message(error: Exception, request) -> str:
    """Turn any tool exception into text the model can read and recover from."""
    return f"error: {request.tool_call['name']} failed: {type(error).__name__}: {error}"


def build_reliability_middleware(fallback_model: BaseChatModel | None = None) -> list:
    middleware = [
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS_PER_RUN, exit_behavior="end"),
        ModelRetryMiddleware(max_retries=2, initial_delay=1.0, backoff_factor=2.0),
        ToolRetryMiddleware(max_retries=1),
        ToolErrorMiddleware(on_error=_tool_error_to_message),
    ]
    if fallback_model is not None:
        middleware.append(ModelFallbackMiddleware(fallback_model))
    return middleware
