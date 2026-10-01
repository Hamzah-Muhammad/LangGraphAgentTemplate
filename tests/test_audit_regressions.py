"""
Regression tests for bugs found in the 2026-09-30 audit. Each test reproduces the exact
failure first seen, so it cannot come back unnoticed.
"""

import asyncio
import io
from contextlib import redirect_stdout
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.orchestration.graph import build_graph, build_simple_agent
from agent.orchestration.patterns.evaluator import build_evaluator_graph
from agent.orchestration.patterns.supervisor import build_supervisor_graph
from agent.prompt.loader import load_system_prompt
from agent.shared.runtime import Context
from agent.shared.usage import UsageCounter, with_counter
from agent.tools.files import read_file
from tests.fakes import fake_model

CTX = Context(user_id="u")


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def cfg(thread):
    return {"configurable": {"thread_id": thread}}


def run(coro):
    return asyncio.run(coro)


def test_gated_tool_inside_a_subagent_still_needs_approval():
    """Was: ask_specialist ran example_tool with no approval (bare create_agent)."""
    model = fake_model(
        {"tool_calls": [{"name": "ask_specialist", "args": {"task": "use example_tool"}}]},
        {"tool_calls": [{"name": "example_tool", "args": {"query": "x"}}]},
        "specialist done",
        "main done",
    )

    async def go():
        graph = await build_graph(model, checkpointer=InMemorySaver(), include_mcp=False)
        out = await graph.ainvoke(user("delegate"), cfg("B"), context=CTX)
        assert "__interrupt__" in out, "gated tool ran inside the subagent without approval"
        request = out["__interrupt__"][0].value["action_requests"][0]
        assert request["name"] == "example_tool"

    run(go())


def test_planner_results_do_not_leak_into_the_next_turn():
    """Was: turn 2's synthesize saw turn 1's worker results too."""
    model = fake_model("w1", "final1", "w2", "final2", structured={"steps": ["only step"]})

    async def go():
        graph = await build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                                  include_mcp=False)
        await graph.ainvoke(user("first"), cfg("C"), context=CTX)
        out = await graph.ainvoke(user("second"), cfg("C"), context=CTX)
        assert out["results"] == ["### only step\nw2"]

    run(go())


def test_evaluator_starts_fresh_each_turn():
    """Was: turn 2 revised turn 1's draft with turn 1's rounds already spent."""
    graph = build_evaluator_graph(
        build_simple_agent(fake_model("a", "b"), []),
        fake_model(structured={"passed": True, "feedback": ""}),
        max_rounds=3, checkpointer=InMemorySaver(),
    )

    async def go():
        await graph.ainvoke(user("first"), cfg("D"), context=CTX)
        out = await graph.ainvoke(user("second"), cfg("D"), context=CTX)
        assert out["rounds"] == 1 and out["messages"][-1].content == "b"

    run(go())


def test_supervisor_hop_cap_is_per_request():
    """Was: once a thread hit the cap, every later request stopped immediately."""
    team = {"r": ("r", build_simple_agent(fake_model(*[f"r{i}" for i in range(9)]), []))}
    boss = fake_model(structured=[{"next": "r", "instruction": "go"},
                                  {"next": "FINISH", "instruction": "answer"}] * 3)
    graph = build_supervisor_graph(team, boss, max_hops=2, checkpointer=InMemorySaver())

    async def go():
        for question in ("q1", "q2", "q3"):
            out = await graph.ainvoke(user(question), cfg("E"), context=CTX)
            assert out["hops"] == 1 and out["messages"][-1].content == "answer"

    run(go())


def test_usage_counts_nested_model_calls():
    """Was: the planner reported 0 tokens because workers' calls are nested."""

    def counted(text):
        return AIMessage(content=text, usage_metadata={
            "input_tokens": 50, "output_tokens": 50, "total_tokens": 100})

    model = fake_model(counted("w1"), counted("w2"), counted("final"),
                       structured={"steps": ["a", "b"]})

    async def go():
        graph = await build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                                  include_mcp=False)
        counter = UsageCounter()
        await graph.ainvoke(user("x"), with_counter(cfg("F"), counter), context=CTX)
        # 2 workers + synthesize carry usage; the planner's structured call is counted too
        assert (counter.input_tokens, counter.output_tokens) == (150, 150)
        assert counter.model_calls >= 3

    run(go())


def test_offloaded_results_are_private_to_their_thread(monkeypatch, tmp_path):
    """Was: one shared folder, so any thread or user could read another's results."""
    monkeypatch.setenv("OFFLOAD_DIR", str(tmp_path))
    (tmp_path / "alice-thread").mkdir()
    (tmp_path / "alice-thread" / "dump-c1.txt").write_text("SECRET", encoding="utf-8")

    def as_thread(thread, name):
        runtime = SimpleNamespace(config={"configurable": {"thread_id": thread}})
        return read_file.func(name=name, runtime=runtime)

    assert "SECRET" in as_thread("alice-thread", "dump-c1.txt")
    assert as_thread("bob-thread", "dump-c1.txt").startswith("error")
    assert as_thread("bob-thread", "../alice-thread/dump-c1.txt").startswith("error")
    assert as_thread("bob-thread", "../../pyproject.toml").startswith("error")


def test_evaluator_generator_is_not_told_about_skills_it_cannot_load():
    """Was: the tool-less generator's prompt said to call load_skill."""
    assert "load_skill" in load_system_prompt()
    assert "load_skill" not in load_system_prompt(with_skills=False)


def test_cli_streams_only_the_answer_node():
    """Was: parallel workers' tokens interleaved with the final answer on screen."""
    import run as cli

    model = fake_model("WORKER-TEXT", "FINAL-TEXT", structured={"steps": ["a"]})

    async def go():
        graph = await build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                                  include_mcp=False)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            await cli.drive(graph, user("x"), cfg("S"), cfg("S"), CTX,
                            cli.STREAM_NODES["planner"])
        return buffer.getvalue()

    printed = run(go())
    assert "FINAL-TEXT" in printed and "WORKER-TEXT" not in printed


def test_cli_prints_answers_that_are_not_streamed():
    """Supervisor answers come from structured output, so the CLI prints them whole."""
    import run as cli

    team = {"r": ("r", build_simple_agent(fake_model("fact"), []))}
    boss = fake_model(structured=[{"next": "r", "instruction": "go"},
                                  {"next": "FINISH", "instruction": "FINAL-ANSWER"}])
    graph = build_supervisor_graph(team, boss, checkpointer=InMemorySaver())

    async def go():
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            await cli.drive(graph, user("x"), cfg("P"), cfg("P"), CTX,
                            cli.STREAM_NODES["supervisor"])
        return buffer.getvalue()

    assert "FINAL-ANSWER" in run(go())
