"""
BLOCK 5: MEMORY

Two kinds, both from LangGraph, both persisted in one SQLite file:

  1. Checkpointer (short-term / per-thread). Saves the full graph state after every
     step, keyed by thread_id. Gives you: resume after a crash, multi-turn chat,
     pause-for-approval, replay. Swap for AsyncPostgresSaver in prod.

  2. Store (long-term / cross-thread). Key-value memory namespaced by user, read and
     written by the `remember` / `recall` tools in tools/memory_tools.py.
     Swap for AsyncPostgresStore in prod. Add an `index=` config for semantic search.

Both are async context managers, so run.py opens them with `async with open_memory()`.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

Memory = tuple[AsyncSqliteSaver, AsyncSqliteStore]


@asynccontextmanager
async def open_memory(path: str | None = None) -> AsyncIterator[Memory]:
    path = path or os.getenv("MEMORY_DB_PATH", "memory.db")
    async with (
        AsyncSqliteSaver.from_conn_string(path) as checkpointer,
        AsyncSqliteStore.from_conn_string(path) as store,
    ):
        await store.setup()
        yield checkpointer, store
