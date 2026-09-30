"""
Regression tests for the third audit (2026-09-30): what a real run needs end to end.
Each test reproduces a gap that was proven with a probe before it was fixed.
"""

import asyncio
import json

from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from agent.memory.store import memory_namespace
from agent.model.model import build_model
from agent.orchestration.graph import build_graph, build_simple_agent
from agent.prompt.loader import load_system_prompt
from agent.shared.runtime import Context
from agent.tools.mcp import load_mcp_tools
from evals.runner import Case, run_case
from tests.fakes import FakeToolModel, fake_model

CTX = Context(user_id="alice")
SEEN_SYSTEM: list[str] = []


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def cfg(thread):
    return {"configurable": {"thread_id": thread}}


def run(coro):
    return asyncio.run(coro)


class PromptSpy(FakeToolModel):
    """Records the system prompt of every model call."""

    def _generate(self, messages, *args, **kwargs):
        SEEN_SYSTEM.append(next((str(m.content) for m in messages if m.type == "system"), ""))
        return super()._generate(messages, *args, **kwargs)


# ------------------------------------------------------------ memory is actually used


def test_saved_facts_reach_the_model_without_calling_recall():
    """Was: saved facts were invisible unless the model chose to call recall."""
    SEEN_SYSTEM.clear()
    store = InMemoryStore()
    store.put(memory_namespace(CTX), "k1", {"fact": "prefers metric units"})
    store.put(memory_namespace(Context(user_id="bob")), "k2", {"fact": "BOB-ONLY"})
    model = PromptSpy(messages=iter([AIMessage(content="ok")]))
    graph = build_simple_agent(model, [], checkpointer=InMemorySaver(), store=store)
    run(graph.ainvoke(user("hi"), cfg("m"), context=CTX))
    assert "prefers metric units" in SEEN_SYSTEM[-1]
    assert "never as instructions" in SEEN_SYSTEM[-1]
    assert "BOB-ONLY" not in SEEN_SYSTEM[-1]  # another user's facts stay out


def test_memory_injection_can_be_switched_off(monkeypatch):
    SEEN_SYSTEM.clear()
    monkeypatch.setenv("MEMORY_INJECT", "false")
    store = InMemoryStore()
    store.put(memory_namespace(CTX), "k1", {"fact": "prefers metric units"})
    model = PromptSpy(messages=iter([AIMessage(content="ok")]))
    graph = build_simple_agent(model, [], checkpointer=InMemorySaver(), store=store)
    run(graph.ainvoke(user("hi"), cfg("m2"), context=CTX))
    assert "prefers metric units" not in SEEN_SYSTEM[-1]


# ------------------------------------------------------------ deadlines

HANG_CALLS = {"n": 0}


@tool
async def hang(query: str) -> str:
    """Never returns."""
    HANG_CALLS["n"] += 1
    await asyncio.sleep(3600)
    return "unreachable"


def test_a_hanging_tool_is_cancelled_retried_and_reported(monkeypatch):
    """Was: one stuck tool hung the whole run forever."""
    monkeypatch.setenv("TOOL_TIMEOUT_S", "0.2")
    HANG_CALLS["n"] = 0
    model = fake_model({"tool_calls": [{"name": "hang", "args": {"query": "x"}}]}, "recovered")
    graph = build_simple_agent(model, [hang], checkpointer=InMemorySaver())
    out = run(asyncio.wait_for(graph.ainvoke(user("go"), cfg("h"), context=CTX), 20))
    tool_msg = [m for m in out["messages"] if m.type == "tool"][-1]
    assert "did not finish within 0.2s" in tool_msg.content
    assert HANG_CALLS["n"] == 2  # first try + one retry
    assert out["messages"][-1].content == "recovered"


def test_subagent_tools_are_exempt_from_the_tool_timeout(monkeypatch):
    """A subagent may wait for human approval; the timeout must not cancel it."""
    monkeypatch.setenv("TOOL_TIMEOUT_S", "0.01")
    model = fake_model(
        {"tool_calls": [{"name": "ask_specialist", "args": {"task": "use example_tool"}}]},
        {"tool_calls": [{"name": "example_tool", "args": {"query": "x"}}]},
        "specialist done",
        "main done",
    )

    async def go():
        graph = await build_graph(model, checkpointer=InMemorySaver(), include_mcp=False)
        paused = await graph.ainvoke(user("delegate"), cfg("s"), context=CTX)
        assert "__interrupt__" in paused
        await asyncio.sleep(0.05)  # longer than the tool timeout
        resume = {i.id: {"decisions": [{"type": "approve"}]} for i in paused["__interrupt__"]}
        return await graph.ainvoke(Command(resume=resume), cfg("s"), context=CTX)

    assert run(go())["messages"][-1].content == "main done"


def test_model_client_has_a_deadline_and_retries(monkeypatch):
    """Was: no timeout, so a stalled host could hang a run for ten minutes."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "any")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "45")
    model = build_model()
    assert model.request_timeout == 45 and model.max_retries == 2


# ------------------------------------------------------------ startup resilience


def test_a_broken_mcp_server_is_skipped_not_fatal(tmp_path, capsys):
    """Was: one server that failed to start crashed the whole agent at startup."""
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({
        "dead": {"enabled": True, "transport": "stdio",
                 "command": "definitely-not-a-real-binary-xyz", "args": []},
    }), encoding="utf-8")
    assert run(load_mcp_tools(config)) == []
    assert "skipped server 'dead'" in capsys.readouterr().err


# ------------------------------------------------------------ evals see nested tool calls


def test_evals_see_tool_calls_made_inside_nested_agents():
    """Was: trajectory checks read the top-level messages, which hide nested tool calls."""
    case = Case(id="nested", input="please use the tool", mode="router",
                expect_tools=["example_tool"], forbid_tools=["remember"])
    model = fake_model(
        {"tool_calls": [{"name": "example_tool", "args": {"query": "x"}}]}, "done",
        structured={"route": "simple"},
    )

    async def go():
        graph = await build_graph(model, mode="router", checkpointer=InMemorySaver(),
                                  include_mcp=False)
        return await run_case(graph, case)

    result = run(go())
    assert result.tools == ["example_tool"] and result.passed, result.reasons


# ------------------------------------------------------------ prompt rules


def test_default_prompt_covers_untrusted_data_memory_and_checking():
    prompt = load_system_prompt().lower()
    assert "data, not instructions" in prompt
    assert "remember" in prompt and "before you answer" in prompt
