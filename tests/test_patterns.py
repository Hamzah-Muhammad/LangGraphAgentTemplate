"""Every workflow pattern and harness feature, driven by scripted fake models."""

import asyncio

import pytest
from langchain.tools import tool
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.orchestration.graph import MODES, build_graph, build_simple_agent, load_system_prompt
from agent.orchestration.patterns.evaluator import build_evaluator_graph
from agent.orchestration.patterns.router import build_router_graph
from agent.orchestration.patterns.supervisor import build_supervisor_graph
from agent.prompt.skills import discover_skills
from agent.shared.runtime import Context
from agent.tools.files import read_file
from agent.tools.skills import load_skill
from tests.fakes import fake_model

CTX = Context(user_id="test")


def cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def user(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


def agent(*script):
    return build_simple_agent(fake_model(*script), [])


# ---------------------------------------------------------------- every mode builds


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_compiles(mode):
    graph = asyncio.run(build_graph(fake_model("x"), mode=mode, include_mcp=False))
    assert graph.get_graph() is not None


# ---------------------------------------------------------------- router


def test_router_rule_match_skips_the_model():
    routes = {"general": ("general", agent("from general")), "deep": ("deep", agent("from deep"))}
    router_model = fake_model(structured={"route": "general"})  # would be wrong if asked
    graph = build_router_graph(routes, router_model, rules=[(r"compare", "deep")],
                               checkpointer=InMemorySaver())
    out = asyncio.run(graph.ainvoke(user("compare a and b"), cfg("r1"), context=CTX))
    assert out["routed_by"] == "rule" and out["route"] == "deep"
    assert out["messages"][-1].content == "from deep"


def test_router_falls_back_to_model_when_no_rule_matches():
    routes = {"general": ("general", agent("from general")), "deep": ("deep", agent("from deep"))}
    graph = build_router_graph(routes, fake_model(structured={"route": "deep"}), rules=[],
                               checkpointer=InMemorySaver())
    out = asyncio.run(graph.ainvoke(user("hello"), cfg("r2"), context=CTX))
    assert out["routed_by"] == "model" and out["messages"][-1].content == "from deep"


def test_router_uses_default_when_classifier_breaks():
    routes = {"general": ("general", agent("from general")), "deep": ("deep", agent("from deep"))}
    broken = fake_model(structured={"route": "not-a-route"})  # fails Literal validation
    graph = build_router_graph(routes, broken, rules=[], default="general",
                               checkpointer=InMemorySaver())
    out = asyncio.run(graph.ainvoke(user("hello"), cfg("r3"), context=CTX))
    assert out["routed_by"] == "default" and out["messages"][-1].content == "from general"


# ---------------------------------------------------------------- evaluator


def test_evaluator_revises_until_the_separate_grader_passes():
    generator = fake_model("draft one", "draft two")
    grader = fake_model(structured=[
        {"passed": False, "feedback": "add the price"},
        {"passed": True, "feedback": "good"},
    ])
    graph = build_evaluator_graph(generator, grader, max_rounds=3, checkpointer=InMemorySaver())
    out = asyncio.run(graph.ainvoke(user("write it"), cfg("e1"), context=CTX))
    assert out["rounds"] == 2 and out["messages"][-1].content == "draft two"


def test_evaluator_escalates_to_a_human_after_max_rounds():
    generator = fake_model("d1", "d2", "d3")
    grader = fake_model(structured={"passed": False, "feedback": "still wrong"})
    graph = build_evaluator_graph(generator, grader, max_rounds=3, checkpointer=InMemorySaver())

    async def go():
        paused = await graph.ainvoke(user("write it"), cfg("e2"), context=CTX)
        interrupt = paused["__interrupt__"][0]
        assert interrupt.value["type"] == "evaluator_escalation" and paused["rounds"] == 3
        out = await graph.ainvoke(
            Command(resume={interrupt.id: {"draft": "human fixed"}}), cfg("e2"), context=CTX
        )
        assert out["messages"][-1].content == "human fixed"

    asyncio.run(go())


# ---------------------------------------------------------------- supervisor


def test_supervisor_delegates_then_finishes():
    team = {
        "researcher": ("finds facts", agent("fact: the sky is blue")),
        "writer": ("writes", agent("The sky is blue.")),
    }
    boss = fake_model(structured=[
        {"next": "researcher", "instruction": "find the sky colour"},
        {"next": "writer", "instruction": "write it up"},
        {"next": "FINISH", "instruction": "The sky is blue."},
    ])
    graph = build_supervisor_graph(team, boss, max_hops=6, checkpointer=InMemorySaver())
    out = asyncio.run(graph.ainvoke(user("what colour is the sky?"), cfg("s1"), context=CTX))
    names = [m.name for m in out["messages"] if m.type == "ai"]
    assert names == ["researcher", "writer", "supervisor"]
    assert out["hops"] == 2 and out["messages"][-1].content == "The sky is blue."


def test_supervisor_stops_at_the_hop_cap():
    team = {"researcher": ("finds facts", agent("r1", "r2", "r3"))}
    boss = fake_model(structured={"next": "researcher", "instruction": "again"})
    graph = build_supervisor_graph(team, boss, max_hops=2, checkpointer=InMemorySaver())
    out = asyncio.run(graph.ainvoke(user("loop forever"), cfg("s2"), context=CTX))
    assert out["hops"] == 2 and out["messages"][-1].content.startswith("Stopped after 2")


# ---------------------------------------------------------------- planner cap


def test_planner_caps_workers_in_code(monkeypatch):
    monkeypatch.setenv("MAX_WORKERS", "3")
    model = fake_model("w1", "w2", "w3", "final",
                       structured={"steps": [f"step {i}" for i in range(10)]})
    graph = asyncio.run(build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                                    include_mcp=False))
    out = asyncio.run(graph.ainvoke(user("compare ten things"), cfg("p1"), context=CTX))
    assert len(out["plan"]) == 3 and len(out["results"]) == 3


# ---------------------------------------------------------------- offload + read_file


@tool
def big_dump(query: str) -> str:
    """Returns a very large result."""
    return "\n".join(f"line {i}" for i in range(5000))


def test_large_tool_results_are_offloaded_and_readable(monkeypatch, tmp_path):
    monkeypatch.setenv("OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("OFFLOAD_CHARS", "1000")
    model = fake_model(
        {"tool_calls": [{"name": "big_dump", "args": {"query": "x"}, "id": "c1"}]},
        {"tool_calls": [{"name": "read_file",
                         "args": {"name": "big_dump-c1.txt", "offset": 4000, "limit": 2},
                         "id": "c2"}]},
        "done",
    )
    graph = build_simple_agent(model, [big_dump, read_file], checkpointer=InMemorySaver())
    out = asyncio.run(graph.ainvoke(user("dump"), cfg("o1"), context=CTX))
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert "saved to big_dump-c1.txt" in tool_msgs[0].content
    assert len(tool_msgs[0].content) < 1500  # the model never saw the 40k chars
    assert (tmp_path / "o1" / "big_dump-c1.txt").exists()  # one folder per thread
    assert "line 4000" in tool_msgs[1].content and "line 4001" in tool_msgs[1].content



# ---------------------------------------------------------------- skills


def test_skills_are_indexed_and_loaded_on_demand():
    assert "write-report" in discover_skills()
    assert "write-report" in load_system_prompt()  # index line only
    assert "Open with the answer" not in load_system_prompt()  # body stays out
    assert "Open with the answer" in load_skill.invoke({"name": "write-report"})
    assert load_skill.invoke({"name": "nope"}).startswith("error")


# ---------------------------------------------------------------- time travel


def test_replay_from_an_earlier_checkpoint():
    async def go():
        graph = build_simple_agent(fake_model("first", "second"), [], checkpointer=InMemorySaver())
        await graph.ainvoke(user("hi"), cfg("tt"), context=CTX)
        history = [s async for s in graph.aget_state_history(cfg("tt"))]
        before_model = next(s for s in history if s.next == ("model",))
        out = await graph.ainvoke(None, before_model.config, context=CTX)
        assert out["messages"][-1].content == "second"  # the model step ran again

    asyncio.run(go())


def test_todo_middleware_is_opt_in(monkeypatch):
    off = build_simple_agent(fake_model("x"), [])
    monkeypatch.setenv("AGENT_TODOS", "true")
    on = build_simple_agent(fake_model("x"), [])
    assert "todos" not in off.channels and "todos" in on.channels


def test_simple_agent_answers_with_ai_message():
    graph = agent("hello")
    out = asyncio.run(graph.ainvoke(user("hi"), context=CTX))
    assert isinstance(out["messages"][-1], AIMessage)
