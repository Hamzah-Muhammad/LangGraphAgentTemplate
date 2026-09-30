"""
BLOCK 5: MEMORY

Two kinds, both from LangGraph:

  1. Checkpointer (short-term / per-thread). Saves the full graph state after every
     step, keyed by thread_id. This is what gives you: resume after a crash, multi-turn
     chat, pause-for-approval, replay. SQLite here; swap for PostgresSaver in prod.

  2. Store (long-term / cross-thread). Key-value memory the agent can read and write
     across conversations (user preferences, facts learned). InMemoryStore here as a
     placeholder; swap for a persistent store when you need it.
"""

import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore


def build_checkpointer() -> SqliteSaver:
    path = os.getenv("MEMORY_DB_PATH", "memory.db")
    # check_same_thread=False: the graph may touch the connection from worker threads.
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_store() -> InMemoryStore:
    # PLACEHOLDER. Persistent option: from langgraph.store.postgres import PostgresStore
    return InMemoryStore()
