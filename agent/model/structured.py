# agent/model/structured.py  |  BLOCK 1 MODEL
"""
STRUCTURED OUTPUT, made portable and robust

Every decision a workflow takes from the model (plan, route, verdict, next specialist)
goes through structured(). Two things it fixes:

  1. Portability. ChatOpenAI's default structured-output method is strict `json_schema`,
     which many OpenAI-compatible hosts (Groq, NIM, local servers, older models) reject.
     Plain tool calling ("function_calling") works anywhere the agent's tools work.
  2. Robustness. Open models sometimes return malformed arguments. One retry fixes most
     of those. Callers still wrap the call and fall back to a safe default, so a bad
     reply degrades the workflow instead of crashing the run.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

STRUCTURED_METHOD = "function_calling"
RETRIES = 1


def structured(model: BaseChatModel, schema: Any) -> Runnable:
    """model.with_structured_output(schema), portable, with one retry on bad output."""
    runnable = model.with_structured_output(schema, method=STRUCTURED_METHOD)
    # No backoff: a malformed reply is not a rate limit, so retry at once. Transient HTTP
    # errors (429, 5xx) are retried with backoff by the ChatOpenAI client (max_retries).
    return runnable.with_retry(stop_after_attempt=RETRIES + 1, wait_exponential_jitter=False)
