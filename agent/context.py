"""
BLOCK 4: CONTEXT WINDOW

What the model sees on each call. Two middleware pieces:

  1. `trim_context`  - a before_model hook. Runs right before every model call.
                       Placeholder: drops nothing. Put your own rules here
                       (cap message count, strip old tool outputs, inject a date).
  2. SummarizationMiddleware - when the history gets long, the framework asks the model
                       to summarize old messages and replaces them with the summary.

Middleware is the LangChain 1.x way to shape the loop without rewriting it.
"""

from langchain.agents.middleware import SummarizationMiddleware, before_model
from langchain.agents.middleware.types import AgentState
from langchain_core.language_models import BaseChatModel
from langgraph.runtime import Runtime

# How many messages to keep before summarizing kicks in. Tune per model.
SUMMARIZE_AFTER_MESSAGES = 40
KEEP_LAST_MESSAGES = 20


@before_model
def trim_context(state: AgentState, runtime: Runtime) -> dict | None:
    """Runs before each model call. Return a state update or None for no change.

    PLACEHOLDER. Example of a real rule:
        msgs = state["messages"]
        if len(msgs) > 60:
            return {"messages": [RemoveMessage(id=m.id) for m in msgs[:20]]}
    """
    return None


def build_context_middleware(model: BaseChatModel) -> list:
    """Middleware list for the Context Window block, in execution order."""
    return [
        trim_context,
        SummarizationMiddleware(
            model,
            trigger=("messages", SUMMARIZE_AFTER_MESSAGES),
            keep=("messages", KEEP_LAST_MESSAGES),
        ),
    ]
