# agent/context/compaction.py  |  BLOCK 4 CONTEXT WINDOW
"""
BLOCK 4: CONTEXT WINDOW

What the model sees on each call. Compaction happens on your terms, well below the
model's limit, never under pressure. Layers, in order:

  1. trim_context              before_model hook. Your own rules (placeholder).
  2. ContextEditingMiddleware  at 50% of the window, clears OLD tool results (keeps the
                               last 3). Cheap: no model call. Most context bloat is tool
                               output the model already acted on.
  3. SummarizationMiddleware   at 60% of the window, summarizes older messages into a
                               structured record and keeps the last 20. Costs one call.
  4. TodoListMiddleware        opt-in (AGENT_TODOS=true). A write_todos tool that does
                               nothing except keep the plan in recent attention.

Large single tool results are handled separately by agent/context/offload.py.
"""

from langchain.agents.middleware import (
    ContextEditingMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    before_model,
)
from langchain.agents.middleware.context_editing import ClearToolUsesEdit
from langchain.agents.middleware.types import AgentState
from langchain_core.language_models import BaseChatModel
from langgraph.runtime import Runtime

from agent.shared.settings import Settings

KEEP_LAST_MESSAGES = 20
KEEP_LAST_TOOL_RESULTS = 3
# Never clear these: their results ARE the working memory.
NEVER_CLEAR = ("write_todos", "load_skill")


@before_model
def trim_context(state: AgentState, runtime: Runtime) -> dict | None:
    """Runs before each model call. Return a state update or None for no change.

    PLACEHOLDER. Example of a real rule:
        msgs = state["messages"]
        if len(msgs) > 60:
            return {"messages": [RemoveMessage(id=m.id) for m in msgs[:20]]}
    """
    return None


def build_context_middleware(model: BaseChatModel, settings: Settings) -> list:
    """Middleware list for the Context Window block, in execution order."""
    window = settings.context_window_tokens
    middleware = [
        trim_context,
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=int(window * 0.5),
                    keep=KEEP_LAST_TOOL_RESULTS,
                    exclude_tools=NEVER_CLEAR,
                )
            ]
        ),
        SummarizationMiddleware(
            model,
            trigger=("tokens", int(window * 0.6)),
            keep=("messages", KEEP_LAST_MESSAGES),
        ),
    ]
    if settings.todos:
        middleware.append(TodoListMiddleware())
    return middleware
