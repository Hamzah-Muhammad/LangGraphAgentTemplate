"""Middleware order is load-bearing. These tests pin the behaviour, not the list."""

import asyncio

from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.orchestration.graph import build_graph, build_simple_agent
from agent.shared.runtime import Context
from tests.fakes import fake_model

CALLS = {"boom": 0}


@tool
def boom(query: str) -> str:
    """Always fails. Exists to test error handling."""
    CALLS["boom"] += 1
    raise RuntimeError("simulated outage")


def test_failing_tool_is_retried_then_reported_to_model():
    CALLS["boom"] = 0
    call = AIMessage(content="", tool_calls=[{"name": "boom", "args": {"query": "x"}, "id": "c1"}])

    async def go():
        agent = build_simple_agent(
            fake_model(call, "recovered"), [boom], checkpointer=InMemorySaver()
        )
        out = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "go"}]},
            config={"configurable": {"thread_id": "r1"}},
        )
        tool_msg = [m for m in out["messages"] if m.type == "tool"][-1]
        assert "simulated outage" in tool_msg.content  # error reached the model as text
        assert out["messages"][-1].content == "recovered"  # and the run continued

    asyncio.run(go())
    assert CALLS["boom"] == 2  # 1 attempt + 1 retry (ToolRetryMiddleware max_retries=1)


def test_planner_parallel_interrupts_resume_by_id_map():
    """Two workers pause at the approval gate at once; one id-keyed resume releases both."""
    call = AIMessage(
        content="", tool_calls=[{"name": "example_tool", "args": {"query": "p"}, "id": "c1"}]
    )

    async def go():
        # each worker: tool call -> (approve) -> answer; then synthesize answers once
        model = fake_model(call, call, "w1", "w2", "final", structured={"steps": ["A", "B"]})
        graph = await build_graph(
            model, mode="planner", checkpointer=InMemorySaver(), include_mcp=False
        )
        config = {"configurable": {"thread_id": "pi"}}
        ctx = Context(user_id="u")
        paused = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "two things"}]}, config=config, context=ctx
        )
        interrupts = paused["__interrupt__"]
        assert len(interrupts) == 2
        resume = {i.id: {"decisions": [{"type": "approve"}]} for i in interrupts}
        out = await graph.ainvoke(Command(resume=resume), config=config, context=ctx)
        assert len(out["results"]) == 2
        assert out["messages"][-1].content == "final"

    asyncio.run(go())
