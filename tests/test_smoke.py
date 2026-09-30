"""
Smoke tests: build the full graph with a scripted fake model and run it.
Proves the six blocks wire together, that a tool call round-trips, and that the
approval gate pauses and resumes. Needs no API key and no network.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.graph import build_graph, load_system_prompt


class FakeToolModel(GenericFakeChatModel):
    """GenericFakeChatModel that accepts bind_tools (returns itself, ignores the tools)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _model(*scripted: AIMessage) -> FakeToolModel:
    return FakeToolModel(messages=iter(scripted))


def _graph(model):
    return build_graph(model, checkpointer=InMemorySaver())


def test_system_prompt_loads():
    assert "Role" in load_system_prompt()


def test_graph_builds_and_runs_one_turn():
    graph = _graph(_model(AIMessage(content="ok from fake model")))
    config = {"configurable": {"thread_id": "t1"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config=config)
    assert result["messages"][-1].content == "ok from fake model"


def test_memory_persists_across_turns():
    graph = _graph(_model(AIMessage(content="one"), AIMessage(content="two")))
    config = {"configurable": {"thread_id": "t2"}}
    graph.invoke({"messages": [{"role": "user", "content": "a"}]}, config=config)
    result = graph.invoke({"messages": [{"role": "user", "content": "b"}]}, config=config)
    assert len(result["messages"]) == 4  # 2 user + 2 ai survived in the thread


def test_tool_call_pauses_for_approval_then_runs():
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "example_tool", "args": {"query": "ping"}, "id": "call_1"}],
    )
    graph = _graph(_model(tool_call, AIMessage(content="done")))
    config = {"configurable": {"thread_id": "t3"}}

    # 1. model asks for the gated tool -> graph pauses before running it
    paused = graph.invoke({"messages": [{"role": "user", "content": "test the tool"}]}, config=config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["action_requests"][0]["name"] == "example_tool"

    # 2. human approves -> tool runs -> model answers
    result = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_msgs and "example_tool received: 'ping'" in tool_msgs[-1].content
    assert result["messages"][-1].content == "done"
