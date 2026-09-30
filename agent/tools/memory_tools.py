# agent/tools/memory_tools.py  |  BLOCK 2 TOOLS
"""
Long-term memory tools. Backed by the LangGraph Store (BLOCK 5), namespaced per user.

The model decides WHEN to remember; the Store decides WHERE it lives. Facts survive
across threads and restarts. `runtime.context.user_id` comes from agent/shared/runtime.py.
"""

from uuid import uuid4

from langchain.tools import ToolRuntime, tool

from agent.shared.runtime import Context


def _namespace(runtime: ToolRuntime) -> tuple[str, str]:
    ctx = runtime.context
    user_id = getattr(ctx, "user_id", None) or Context().user_id
    return ("memories", user_id)


@tool
async def remember(fact: str, runtime: ToolRuntime[Context]) -> str:
    """Save a durable fact about the user or their preferences for future conversations.
    Use only for things worth keeping long-term (name, preferences, standing rules)."""
    if runtime.store is None:
        return "error: no store configured"
    await runtime.store.aput(_namespace(runtime), str(uuid4()), {"fact": fact})
    return f"remembered: {fact}"


@tool
async def recall(topic: str, runtime: ToolRuntime[Context]) -> str:
    """Look up previously saved facts about the user related to a topic.
    Use at the start of a task when past preferences might matter."""
    if runtime.store is None:
        return "error: no store configured"
    # Without an `index=` on the store this is a plain listing (limit 20); add an index
    # config in agent/memory/store.py to make `query=` semantic.
    items = await runtime.store.asearch(_namespace(runtime), query=topic, limit=20)
    if not items:
        return "no saved facts"
    return "\n".join(f"- {item.value['fact']}" for item in items)
