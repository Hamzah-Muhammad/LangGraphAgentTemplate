"""
Smoke tests: build the graphs with a scripted fake model and run them.
Proves the six blocks wire together, a tool call round-trips, the approval gate pauses
and resumes, long-term memory persists, and the planner fans out. No key, no network.
"""

import asyncio

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from agent.orchestration.graph import build_graph, load_system_prompt
from agent.shared.runtime import Context
from tests.fakes import fake_model


def run(coro):
    return asyncio.run(coro)


async def _graph(model, mode="simple", store=None):
    return await build_graph(
        model, mode=mode, checkpointer=InMemorySaver(), store=store, include_mcp=False
    )


def test_system_prompt_loads():
    assert "Role" in load_system_prompt()


def test_graph_builds_and_runs_one_turn():
    async def go():
        graph = await _graph(fake_model("ok from fake model"))
        config = {"configurable": {"thread_id": "t1"}}
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "hi"}]}, config=config
        )
        assert result["messages"][-1].content == "ok from fake model"

    run(go())


def test_memory_persists_across_turns():
    async def go():
        graph = await _graph(fake_model("one", "two"))
        config = {"configurable": {"thread_id": "t2"}}
        await graph.ainvoke({"messages": [{"role": "user", "content": "a"}]}, config=config)
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "b"}]}, config=config
        )
        assert len(result["messages"]) == 4  # 2 user + 2 ai survived in the thread

    run(go())


def test_tool_call_pauses_for_approval_then_runs():
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "example_tool", "args": {"query": "ping"}, "id": "call_1"}],
    )

    async def go():
        graph = await _graph(fake_model(tool_call, "done"))
        config = {"configurable": {"thread_id": "t3"}}
        paused = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "test the tool"}]}, config=config
        )
        assert "__interrupt__" in paused
        req = paused["__interrupt__"][0].value["action_requests"][0]
        assert req["name"] == "example_tool"

        result = await graph.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config=config
        )
        tool_msgs = [m for m in result["messages"] if m.type == "tool"]
        assert tool_msgs and "example_tool received: 'ping'" in tool_msgs[-1].content
        assert result["messages"][-1].content == "done"

    run(go())


def test_long_term_memory_tools_round_trip_per_user():
    remember_call = AIMessage(
        content="",
        tool_calls=[{"name": "remember", "args": {"fact": "likes tea"}, "id": "c1"}],
    )
    recall_call = AIMessage(
        content="",
        tool_calls=[{"name": "recall", "args": {"topic": "drinks"}, "id": "c2"}],
    )

    async def go():
        store = InMemoryStore()
        alice = Context(user_id="alice")
        g1 = await _graph(fake_model(remember_call, "saved"), store=store)
        await g1.ainvoke(
            {"messages": [{"role": "user", "content": "remember I like tea"}]},
            config={"configurable": {"thread_id": "m1"}},
            context=alice,
        )
        # new thread, same user, same store -> fact is visible
        g2 = await _graph(fake_model(recall_call, "ok"), store=store)
        out = await g2.ainvoke(
            {"messages": [{"role": "user", "content": "what do I drink?"}]},
            config={"configurable": {"thread_id": "m2"}},
            context=alice,
        )
        tool_msg = [m for m in out["messages"] if m.type == "tool"][-1]
        assert "likes tea" in tool_msg.content
        # different user -> nothing
        g3 = await _graph(fake_model(recall_call, "ok"), store=store)
        out = await g3.ainvoke(
            {"messages": [{"role": "user", "content": "what do I drink?"}]},
            config={"configurable": {"thread_id": "m3"}},
            context=Context(user_id="bob"),
        )
        assert "no saved facts" in [m for m in out["messages"] if m.type == "tool"][-1].content

    run(go())


def test_planner_fans_out_and_synthesizes():
    async def go():
        # plan -> 2 workers (each answers once) -> synthesize (answers once)
        model = fake_model("w1", "w2", "final", structured={"steps": ["task A", "task B"]})
        graph = await _graph(model, mode="planner")
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "do two things"}]},
            config={"configurable": {"thread_id": "p1"}},
            context=Context(user_id="x"),
        )
        assert result["plan"] == ["task A", "task B"]
        assert len(result["results"]) == 2
        assert result["messages"][-1].content == "final"

    run(go())
