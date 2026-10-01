# agent/tools/memory_tools.py  |  BLOCK 2 TOOLS
"""
Long-term memory tools. Backed by the LangGraph Store (BLOCK 5), namespaced per user.

The model decides WHEN to remember; the Store decides WHERE it lives. Facts survive
across threads and restarts. The user comes from `resolve_user_id` in agent/memory/store.py.

Saved facts are read back into the system prompt (agent/memory/inject.py), so these tools
are a write path into the prompt. They are bounded in code: a fact has a length cap, the
same fact is stored once, and MEMORY_REQUIRE_APPROVAL=true puts a human in front of both
`remember` and `forget` (agent/orchestration/approval.py).
"""

import hashlib

from langchain.tools import ToolRuntime, tool

from agent.memory.store import memory_namespace
from agent.shared.runtime import Context
from agent.shared.settings import get_settings

SCAN_LIMIT = 200  # facts looked at by forget


def _namespace(runtime: ToolRuntime) -> tuple[str, str]:
    return memory_namespace(runtime.context, runtime.config)


def _key(fact: str) -> str:
    """Same fact (ignoring case and spacing) -> same key, so saving it twice is a no-op."""
    return hashlib.sha256(" ".join(fact.lower().split()).encode()).hexdigest()[:32]


@tool
async def remember(fact: str, runtime: ToolRuntime[Context]) -> str:
    """Save a durable fact about the user or their preferences for future conversations.
    Use only for things worth keeping long-term (name, preferences, standing rules)."""
    if runtime.store is None:
        return "error: no store configured"
    fact = " ".join(fact.split())
    limit = get_settings().memory_fact_max_chars
    if not fact:
        return "error: nothing to remember"
    if len(fact) > limit:
        return f"error: fact is {len(fact)} characters, the limit is {limit}. Shorten it."
    namespace, key = _namespace(runtime), _key(fact)
    if await runtime.store.aget(namespace, key) is not None:
        return f"already remembered: {fact}"
    await runtime.store.aput(namespace, key, {"fact": fact})
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


@tool
async def forget(topic: str, runtime: ToolRuntime[Context]) -> str:
    """Delete saved facts about the user that contain the given words. Use when the user
    asks you to forget something or says a saved fact is wrong."""
    if runtime.store is None:
        return "error: no store configured"
    needle = " ".join(topic.lower().split())
    if len(needle) < 3:
        return "error: give at least 3 characters so one call cannot wipe everything"
    namespace = _namespace(runtime)
    removed = []
    for item in await runtime.store.asearch(namespace, limit=SCAN_LIMIT):
        fact = item.value.get("fact", "")
        if needle in " ".join(fact.lower().split()):
            await runtime.store.adelete(namespace, item.key)
            removed.append(fact)
    if not removed:
        return "no saved fact matched"
    return "forgot:\n" + "\n".join(f"- {fact}" for fact in removed)
