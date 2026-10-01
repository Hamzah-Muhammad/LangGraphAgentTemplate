# agent/memory/store.py  |  BLOCK 5 MEMORY
"""
BLOCK 5: MEMORY

Two kinds, both from LangGraph, both persisted in one SQLite file:

  1. Checkpointer (short-term / per-thread). Saves the full graph state after every
     step, keyed by thread_id. Gives you: resume after a crash, multi-turn chat,
     pause-for-approval, replay. Swap for AsyncPostgresSaver in prod.

  2. Store (long-term / cross-thread). Key-value memory namespaced by user, read and
     written by the `remember` / `recall` tools in agent/tools/memory_tools.py.
     Swap for AsyncPostgresStore in prod. Add an `index=` config for semantic search.

Both are async context managers, so run.py opens them with `async with open_memory()`.
"""

import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

Memory = tuple[AsyncSqliteSaver, AsyncSqliteStore]


ANONYMOUS = "anonymous"


def resolve_user_id(context, config=None) -> str:
    """Who is this run for? First match wins:
      1. context.user_id            (run.py --user, evals, your own code)
      2. the authenticated user     (LangGraph server with an auth handler: its identity)
      3. config["configurable"]["user_id"]   (a server client that sends it per run)
      4. "anonymous"
    Without 2 or 3 a server would put every caller in one shared memory."""
    user = getattr(context, "user_id", None)
    if user and user != ANONYMOUS:
        return str(user)
    configurable = (config or {}).get("configurable") or {}
    auth = configurable.get("langgraph_auth_user")
    identity = auth.get("identity") if isinstance(auth, dict) else getattr(auth, "identity", None)
    return str(identity or configurable.get("user_id") or ANONYMOUS)


def memory_namespace(context, config=None) -> tuple[str, str]:
    """Where one user's long-term facts live. Used by the memory tools and by inject.py.
    Store labels cannot hold dots (an email would break), so they become underscores."""
    user = re.sub(r"[^A-Za-z0-9_@+-]", "_", resolve_user_id(context, config))[:120]
    return ("memories", user)


@asynccontextmanager
async def open_memory(path: str | None = None) -> AsyncIterator[Memory]:
    path = path or os.getenv("MEMORY_DB_PATH", "memory.db")
    async with (
        AsyncSqliteSaver.from_conn_string(path) as checkpointer,
        AsyncSqliteStore.from_conn_string(path) as store,
    ):
        await store.setup()
        yield checkpointer, store
