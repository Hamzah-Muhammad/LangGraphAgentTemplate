"""
Hardening round (2026-10-01): who a memory belongs to, bounded memory tools, and the
model-call cap as a setting. No key, no network.
"""

import asyncio
import time

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agent.memory.store import memory_namespace, resolve_user_id
from agent.orchestration.approval import build_approval_middleware
from agent.orchestration.graph import build_simple_agent
from agent.orchestration.reliability import build_reliability_middleware
from agent.shared.runtime import Context
from agent.tools import ALL_TOOLS
from agent.tools.memory_tools import forget, remember
from tests.fakes import fake_model
from tests.test_run_hardening import SEEN_SYSTEM, PromptSpy

ALICE = Context(user_id="alice")


def run(coro):
    return asyncio.run(coro)


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def cfg(thread, **extra):
    return {"configurable": {"thread_id": thread, **extra}}


def call(name, **args):
    return {"tool_calls": [{"name": name, "args": args}]}


def facts(store, user_id="alice"):
    return sorted(i.value["fact"] for i in run(store.asearch(("memories", user_id), limit=50)))


# ------------------------------------------------------------ who the run is for


def test_context_user_wins_over_config():
    config = cfg("t", user_id="from-config")
    assert resolve_user_id(Context(user_id="alice"), config) == "alice"


def test_server_run_uses_the_authenticated_identity_then_the_config_user():
    anonymous = Context()
    assert resolve_user_id(anonymous, cfg("t", langgraph_auth_user={"identity": "u-1"})) == "u-1"
    assert resolve_user_id(anonymous, cfg("t", user_id="u-2")) == "u-2"
    assert resolve_user_id(anonymous, cfg("t")) == "anonymous"
    assert resolve_user_id(None, None) == "anonymous"


def test_namespace_is_safe_for_the_store():
    assert memory_namespace(Context(user_id="a.b@x.com")) == ("memories", "a_b@x_com")


def test_server_users_do_not_share_memory():
    """Was: every server caller landed in the 'anonymous' namespace."""
    SEEN_SYSTEM.clear()
    store = InMemoryStore()
    run(store.aput(("memories", "u-1"), "k", {"fact": "U1-ONLY"}))
    model = PromptSpy(messages=iter([AIMessage(content="ok"), AIMessage(content="ok")]))
    graph = build_simple_agent(model, [], checkpointer=InMemorySaver(), store=store)
    run(graph.ainvoke(user("hi"), cfg("a", langgraph_auth_user={"identity": "u-1"}),
                      context=Context()))
    run(graph.ainvoke(user("hi"), cfg("b", langgraph_auth_user={"identity": "u-2"}),
                      context=Context()))
    assert "U1-ONLY" in SEEN_SYSTEM[0]
    assert "U1-ONLY" not in SEEN_SYSTEM[1]


# ------------------------------------------------------------ memory is bounded


def agent_with(*script, store):
    return build_simple_agent(
        fake_model(*script), [remember, forget], checkpointer=InMemorySaver(), store=store
    )


def test_the_same_fact_is_stored_once():
    store = InMemoryStore()
    graph = agent_with(call("remember", fact="Likes tea"), call("remember", fact="  likes   TEA "),
                       "done", store=store)
    run(graph.ainvoke(user("x"), cfg("d"), context=ALICE))
    assert facts(store) == ["Likes tea"]


def test_a_fact_over_the_limit_is_refused(monkeypatch):
    monkeypatch.setenv("MEMORY_FACT_MAX_CHARS", "20")
    store = InMemoryStore()
    graph = agent_with(call("remember", fact="x" * 21), "done", store=store)
    result = run(graph.ainvoke(user("x"), cfg("e"), context=ALICE))
    assert facts(store) == []
    assert "limit is 20" in str(result["messages"][-2].content)


def test_forget_removes_matches_and_refuses_a_tiny_needle():
    store = InMemoryStore()
    for key, fact in [("1", "likes tea"), ("2", "likes coffee"), ("3", "lives in Toronto")]:
        run(store.aput(("memories", "alice"), key, {"fact": fact}))
    graph = agent_with(call("forget", topic="li"), call("forget", topic="tea"), "done",
                       store=store)
    result = run(graph.ainvoke(user("x"), cfg("f"), context=ALICE))
    assert facts(store) == ["likes coffee", "lives in Toronto"]
    replies = [str(m.content) for m in result["messages"] if m.type == "tool"]
    assert "at least 3" in replies[0] and "likes tea" in replies[1]


def test_forget_cannot_touch_another_users_facts():
    store = InMemoryStore()
    run(store.aput(("memories", "bob"), "1", {"fact": "likes tea"}))
    graph = agent_with(call("forget", topic="tea"), "done", store=store)
    run(graph.ainvoke(user("x"), cfg("g"), context=ALICE))
    assert facts(store, "bob") == ["likes tea"]


def test_injection_prefers_the_newest_facts(monkeypatch):
    SEEN_SYSTEM.clear()
    monkeypatch.setenv("MEMORY_INJECT_LIMIT", "2")
    store = InMemoryStore()
    for n in range(5):
        run(store.aput(("memories", "alice"), str(n), {"fact": f"fact-{n}"}))
        time.sleep(0.02)  # Windows clocks are coarse; keep updated_at distinct
    model = PromptSpy(messages=iter([AIMessage(content="ok")]))
    graph = build_simple_agent(model, [], checkpointer=InMemorySaver(), store=store)
    run(graph.ainvoke(user("hi"), cfg("h"), context=ALICE))
    assert "fact-4" in SEEN_SYSTEM[0] and "fact-3" in SEEN_SYSTEM[0]
    assert "fact-0" not in SEEN_SYSTEM[0]


def test_memory_tools_can_require_approval(monkeypatch):
    assert "remember" not in build_approval_middleware().interrupt_on
    monkeypatch.setenv("MEMORY_REQUIRE_APPROVAL", "true")
    gated = build_approval_middleware().interrupt_on
    assert "remember" in gated and "forget" in gated


def test_forget_is_registered():
    assert forget in ALL_TOOLS


# ------------------------------------------------------------ the call cap is a setting


def test_model_call_cap_comes_from_settings(monkeypatch):
    monkeypatch.setenv("MAX_MODEL_CALLS", "7")
    limit = build_reliability_middleware()[0]
    assert limit.run_limit == 7
