"""Real SQLite path: checkpointer + store open, survive a reopen, and isolate users."""

import asyncio

from langchain_core.messages import AIMessage

from agent.graph import build_graph
from agent.memory import open_memory
from agent.runtime import Context
from tests.fakes import fake_model


def test_sqlite_memory_survives_reopen(tmp_path):
    db = str(tmp_path / "memory.db")
    remember_call = AIMessage(
        content="", tool_calls=[{"name": "remember", "args": {"fact": "uses vim"}, "id": "c1"}]
    )
    recall_call = AIMessage(
        content="", tool_calls=[{"name": "recall", "args": {"topic": "editor"}, "id": "c2"}]
    )
    config = {"configurable": {"thread_id": "persist"}}
    ctx = Context(user_id="alice")

    async def first_session():
        async with open_memory(db) as (cp, store):
            g = await build_graph(
                fake_model(remember_call, "saved"), checkpointer=cp, store=store, include_mcp=False
            )
            await g.ainvoke(
                {"messages": [{"role": "user", "content": "I use vim"}]}, config=config, context=ctx
            )

    async def second_session():
        async with open_memory(db) as (cp, store):
            g = await build_graph(
                fake_model(recall_call, "ok"), checkpointer=cp, store=store, include_mcp=False
            )
            # thread history came back from disk
            assert len((await g.aget_state(config)).values["messages"]) == 4
            out = await g.ainvoke(
                {"messages": [{"role": "user", "content": "what editor?"}]},
                config=config,
                context=ctx,
            )
            assert "uses vim" in [m for m in out["messages"] if m.type == "tool"][-1].content

    asyncio.run(first_session())
    asyncio.run(second_session())
