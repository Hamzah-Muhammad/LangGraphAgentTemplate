"""
The Claude subscription adapter (agent/model/claude_code.py), tested without the CLI:
every test replaces `sdk_call`, the single seam between the adapter and the Agent SDK.
A real run is a manual step: MODEL_PROVIDER=claude-code python -m evals
"""

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.model import claude_code
from agent.model.claude_code import ChatClaudeCode, ClaudeCodeError, render_conversation
from agent.model.model import build_fallback_model, build_grader_model, build_model
from agent.model.structured import structured
from agent.orchestration.graph import build_graph, build_simple_agent
from agent.shared.runtime import Context
from agent.shared.schemas import Plan
from agent.shared.usage import UsageCounter, with_counter
from agent.tools.example_tool import example_tool

CTX = Context(user_id="u")
USAGE = {"input_tokens": 100, "cache_read_input_tokens": 20, "output_tokens": 30}


def run(coro):
    return asyncio.run(coro)


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def fake_sdk(monkeypatch, *replies):
    """Replace the SDK seam. Each reply: str (text), list (tool requests the CLI
    deferred back to us) or dict (structured output)."""
    calls, queue = [], list(replies)

    async def fake(prompt, system, model, schema, tools=None):
        calls.append({"prompt": prompt, "system": system, "model": model, "schema": schema,
                      "tools": tools})
        reply = queue.pop(0)
        base = {"text": "", "structured": None, "usage": USAGE, "tool_calls": []}
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            return {**base, "tool_calls": reply}
        if isinstance(reply, dict):
            return {**base, "structured": reply}
        return {**base, "text": reply}

    monkeypatch.setattr(claude_code, "sdk_call", fake)
    return calls


# ------------------------------------------------------------ provider switch


def test_provider_switch_needs_no_api_key(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_PROVIDER", "claude-code")
    monkeypatch.setenv("CLAUDE_MODEL", "haiku")
    model = build_model()
    assert isinstance(model, ChatClaudeCode) and model.model == "haiku"


def test_roles_can_mix_providers(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "claude-code")
    monkeypatch.setenv("GRADER_PROVIDER", "claude-code")
    monkeypatch.setenv("GRADER_MODEL_NAME", "opus")
    monkeypatch.delenv("FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("FALLBACK_API_KEY", raising=False)
    assert build_grader_model().model == "opus"
    assert build_fallback_model() is None  # openai fallback, not configured


def test_unknown_provider_is_a_clear_error(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "banana")
    with pytest.raises(RuntimeError, match="not supported"):
        build_model()


# ------------------------------------------------------------ isolation


def test_claude_code_runs_isolated_with_its_own_tools_off():
    options = claude_code.build_options("haiku", "sys", {"type": "object"})
    assert options.tools == [] and options.mcp_servers == {}
    assert options.strict_mcp_config is True and options.setting_sources == []
    assert options.output_format["type"] == "json_schema"
    assert claude_code.build_options("haiku", "sys", None).max_turns == 1


def test_template_tools_reach_claude_as_deferred_stubs():
    spec = ChatClaudeCode().bind_tools([example_tool]).bound_tools
    options = claude_code.build_options("haiku", "sys", None, spec)
    assert options.tools == []  # still none of Claude Code's own tools
    assert list(options.mcp_servers) == ["host"]
    assert options.allowed_tools == ["mcp__host__example_tool"]
    hook = options.hooks["PreToolUse"][0].hooks[0]
    decision = asyncio.run(hook({}, "id", None))["hookSpecificOutput"]["permissionDecision"]
    assert decision == "defer"  # every tool request is handed back, never executed here
    assert options.max_turns == 1


# ------------------------------------------------------------ message mapping


def test_lone_user_message_passes_through_and_history_becomes_a_transcript():
    system, prompt = render_conversation([SystemMessage("rules"), HumanMessage("hello")])
    assert (system, prompt) == ("rules", "hello")

    _, prompt = render_conversation([
        HumanMessage("find x"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "c1"}]),
        ToolMessage(content="x is 42", tool_call_id="c1", name="lookup"),
    ])
    assert "User: find x" in prompt and "requested tool lookup" in prompt
    assert "Tool result (lookup, id c1):\nx is 42" in prompt


def test_text_reply_carries_usage(monkeypatch):
    fake_sdk(monkeypatch, "Hello!")
    reply = run(ChatClaudeCode(model="haiku").ainvoke("Say hello"))
    assert reply.content == "Hello!" and not reply.tool_calls
    assert reply.usage_metadata["input_tokens"] == 120  # cache tokens are counted
    assert reply.usage_metadata["output_tokens"] == 30


def test_tool_request_becomes_a_langchain_tool_call(monkeypatch):
    calls = fake_sdk(monkeypatch, [
        {"name": "example_tool", "args": {"query": "ping"}, "id": "toolu_1"},
        {"name": "example_tool", "args": {"query": "pong"}, "id": "toolu_2"},
    ])
    model = ChatClaudeCode().bind_tools([example_tool])
    reply = run(model.ainvoke("use the tool twice"))
    assert [c["args"]["query"] for c in reply.tool_calls] == ["ping", "pong"]  # parallel calls
    assert reply.tool_calls[0]["id"] == "toolu_1" and reply.content == ""
    sent = calls[0]
    assert sent["tools"][0]["function"]["name"] == "example_tool"
    assert sent["schema"] is None  # native tool calling, not JSON emulation


def test_structured_helper_works_with_the_adapter(monkeypatch):
    calls = fake_sdk(monkeypatch, {"steps": ["a", "b"]})
    plan = run(structured(ChatClaudeCode(), Plan).ainvoke([HumanMessage("plan it")]))
    assert plan.steps == ["a", "b"]
    assert calls[0]["schema"]["properties"]["steps"]["type"] == "array"


def test_structured_calls_are_counted_in_usage(monkeypatch):
    """Was: plan and review calls bypassed callbacks, so a real planner run reported
    2 model calls when it made 4."""
    fake_sdk(monkeypatch, {"steps": ["a"]})
    counter = UsageCounter()
    run(structured(ChatClaudeCode(), Plan).ainvoke(
        [HumanMessage("plan it")], config={"callbacks": [counter]}))
    assert counter.model_calls == 1 and counter.input_tokens == 120


def test_cli_errors_surface_as_exceptions(monkeypatch):
    fake_sdk(monkeypatch, ClaudeCodeError("Reached maximum number of turns"))
    with pytest.raises(ClaudeCodeError, match="maximum number of turns"):
        run(ChatClaudeCode().ainvoke("x"))


# ------------------------------------------------------------ the template stays in charge


def test_full_tool_loop_keeps_the_approval_gate_and_counts_usage(monkeypatch):
    fake_sdk(
        monkeypatch,
        [{"name": "example_tool", "args": {"query": "ping"}, "id": "toolu_1"}],
        "It returned ping.",
    )
    graph = build_simple_agent(ChatClaudeCode(model="haiku"), [example_tool],
                               checkpointer=InMemorySaver())
    counter = UsageCounter()
    config = with_counter({"configurable": {"thread_id": "cc"}}, counter)

    async def go():
        paused = await graph.ainvoke(user("use example_tool with ping"), config, context=CTX)
        assert paused["__interrupt__"][0].value["action_requests"][0]["name"] == "example_tool"
        resume = {i.id: {"decisions": [{"type": "approve"}]} for i in paused["__interrupt__"]}
        return await graph.ainvoke(Command(resume=resume), config, context=CTX)

    out = run(go())
    tool_msg = [m for m in out["messages"] if m.type == "tool"][-1]
    assert "example_tool received: 'ping'" in tool_msg.content  # the TEMPLATE ran the tool
    assert out["messages"][-1].content == "It returned ping."
    assert counter.model_calls == 2 and counter.input_tokens == 240


def test_every_mode_builds_on_the_adapter(monkeypatch):
    fake_sdk(monkeypatch)
    for mode in ("simple", "planner", "router", "evaluator", "supervisor"):
        graph = run(build_graph(ChatClaudeCode(), mode=mode, include_mcp=False))
        assert graph.get_graph() is not None
