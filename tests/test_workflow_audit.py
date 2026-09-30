"""
Regression tests for the agentic-workflow audit (2026-09-30). Each one pins a behaviour
the research calls for and the template was missing or got wrong.
"""

import asyncio

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from agent.orchestration.graph import build_graph, build_simple_agent
from agent.orchestration.patterns.evaluator import build_evaluator_graph
from agent.orchestration.patterns.router import build_router_graph
from agent.orchestration.patterns.supervisor import build_supervisor_graph
from agent.shared.runtime import Context
from tests.fakes import FakeToolModel, fake_model

CTX = Context(user_id="u")


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


def cfg(thread):
    return {"configurable": {"thread_id": thread}}


def run(coro):
    return asyncio.run(coro)


class BrokenStructured(FakeToolModel):
    """Every structured-output call fails, the way open models sometimes do."""

    def with_structured_output(self, schema, **kwargs):
        def fail(_input):
            raise ValueError("model returned invalid arguments")

        return RunnableLambda(fail)


SPY_CALLS: list = []  # kwargs of each with_structured_output() call
SPY_PROMPTS: list = []  # prompt of each structured call, in the order they ran
SPY_ANSWERS: list = []  # one shared queue: answers go out in call order, whatever schema


class Spy(FakeToolModel):
    """Records every structured-output call and answers from one ordered queue."""

    def with_structured_output(self, schema, **kwargs):
        SPY_CALLS.append(kwargs)

        def answer(messages):
            SPY_PROMPTS.append(" | ".join(str(getattr(m, "content", m)) for m in messages))
            return schema(**SPY_ANSWERS.pop(0))

        return RunnableLambda(answer)


def spy(*script, structured):
    SPY_CALLS.clear()
    SPY_PROMPTS.clear()
    SPY_ANSWERS[:] = list(structured)
    return Spy(messages=iter([AIMessage(content=t) for t in script]))


# ------------------------------------------------------------ portable structured output


def test_structured_output_uses_tool_calling_not_strict_json_schema():
    """Was: ChatOpenAI's default strict json_schema, rejected by many compatible hosts."""
    model = spy("x", structured=[])
    run(build_graph(model, mode="planner", include_mcp=False))
    assert SPY_CALLS and all(c.get("method") == "function_calling" for c in SPY_CALLS)


# ------------------------------------------------------------ malformed output degrades


def test_planner_survives_a_malformed_plan():
    """Was: one bad structured reply crashed the whole run."""
    model = BrokenStructured(messages=iter([AIMessage(content=t) for t in ["w", "final"]]))
    graph = run(build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                            include_mcp=False))
    out = run(graph.ainvoke(user("do it"), cfg("p"), context=CTX))
    assert len(out["results"]) == 1 and out["messages"][-1].content == "final"


def test_evaluator_escalates_when_the_grader_breaks():
    generator = build_simple_agent(fake_model("draft"), [])
    graph = build_evaluator_graph(generator, BrokenStructured(messages=iter([])),
                                  checkpointer=InMemorySaver())
    out = run(graph.ainvoke(user("write"), cfg("e"), context=CTX))
    assert out["__interrupt__"][0].value["message"].startswith("The grader could not")
    assert out["rounds"] == 1  # escalated at once, did not burn the other rounds


def test_supervisor_ends_cleanly_when_its_decision_breaks():
    team = {"r": ("r", build_simple_agent(fake_model("fact"), []))}
    graph = build_supervisor_graph(team, BrokenStructured(messages=iter([])),
                                   checkpointer=InMemorySaver())
    out = run(graph.ainvoke(user("q"), cfg("s"), context=CTX))
    assert "could not coordinate" in out["messages"][-1].content


# ------------------------------------------------------------ multi-turn context


def test_planner_sees_the_earlier_turn_on_a_follow_up():
    """Was: the plan prompt held only the latest message, so follow-ups were meaningless."""
    model = spy("w", "ZANZIBAR wins.", "w", "final",
                structured=[{"steps": ["a"]}, {"done": True}, {"steps": ["b"]}, {"done": True}])
    graph = run(build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                            include_mcp=False))
    run(graph.ainvoke(user("compare ZANZIBAR and QUOKKA"), cfg("m"), context=CTX))
    run(graph.ainvoke(user("now do the same for the third option"), cfg("m"), context=CTX))
    turn_two_plan = SPY_PROMPTS[2]  # plan 1, review 1, plan 2
    assert not SPY_ANSWERS  # every scripted answer used exactly once, no retries
    assert "ZANZIBAR" in turn_two_plan and "third option" in turn_two_plan


def test_router_classifier_sees_the_earlier_turn():
    routes = {"a": ("a", build_simple_agent(fake_model("A1", "A2"), [])),
              "b": ("b", build_simple_agent(fake_model("B1"), []))}
    model = spy(structured=[{"route": "a"}, {"route": "b"}])
    graph = build_router_graph(routes, model, rules=[], checkpointer=InMemorySaver())
    run(graph.ainvoke(user("tell me about ZANZIBAR"), cfg("r"), context=CTX))
    run(graph.ainvoke(user("and the other one?"), cfg("r"), context=CTX))
    assert "ZANZIBAR" in SPY_PROMPTS[-1]


# ------------------------------------------------------------ orchestrator review loop


def test_orchestrator_reviews_and_sends_workers_after_gaps():
    """Was: plan once, synthesize. Now: review results, fill gaps, capped rounds."""
    model = spy("w1", "w2", "final", structured=[
        {"steps": ["first"]},
        {"done": False, "missing_steps": ["gap"]},  # round 1 review: one gap
    ])
    graph = run(build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                            include_mcp=False))
    out = run(graph.ainvoke(user("research"), cfg("o"), context=CTX))
    headers = [r.splitlines()[0] for r in out["results"]]
    assert headers == ["### first", "### gap"]
    assert out["rounds"] == 2 and out["messages"][-1].content == "final"
    assert not SPY_ANSWERS  # round 2 hit the cap, so no second review call


def test_review_loop_is_capped(monkeypatch):
    monkeypatch.setenv("MAX_PLAN_ROUNDS", "2")
    always_gaps = [{"steps": ["s"]}] + [{"done": False, "missing_steps": ["more"]}] * 5
    model = spy(*[f"w{i}" for i in range(6)], "final", structured=always_gaps)
    graph = run(build_graph(model, mode="planner", checkpointer=InMemorySaver(),
                            include_mcp=False))
    out = run(graph.ainvoke(user("research"), cfg("c"), context=CTX))
    assert out["rounds"] == 2 and len(out["results"]) == 2


# ------------------------------------------------------------ concurrency


def test_planner_limits_simultaneous_workers(monkeypatch):
    """Was: every worker hit the model at once, which rate-limits free hosts."""
    monkeypatch.setenv("MAX_CONCURRENCY", "2")
    active = {"now": 0, "peak": 0}

    class Slow(FakeToolModel):
        async def ainvoke(self, *args, **kwargs):
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
            await asyncio.sleep(0.05)
            active["now"] -= 1
            return AIMessage(content="done")

        def with_structured_output(self, schema, **kwargs):
            payload = {"steps": [f"s{i}" for i in range(6)]} if schema.__name__ == "Plan" \
                else {"done": True}
            return RunnableLambda(lambda _input: schema(**payload))

    async def go():
        graph = await build_graph(Slow(messages=iter([])), mode="planner",
                                  checkpointer=InMemorySaver(), include_mcp=False)
        await graph.ainvoke(user("x"), cfg("k"), context=CTX)

    run(go())
    assert active["peak"] == 2


# ------------------------------------------------------------ agentic evaluator


def test_evaluator_writer_can_use_tools():
    """Was: the writer was a bare model with no tools."""
    writer = fake_model(
        {"tool_calls": [{"name": "load_skill", "args": {"name": "write-report"}}]},
        "drafted with the skill",
    )

    async def go():
        graph = await build_graph(writer, mode="evaluator", checkpointer=InMemorySaver(),
                                  include_mcp=False)
        return graph

    # the grader is the same fake model; give it a passing verdict
    writer.structured = {"passed": True, "feedback": ""}
    graph = run(go())
    out = run(graph.ainvoke(user("write a report"), cfg("w"), context=CTX))
    assert out["messages"][-1].content == "drafted with the skill"


# ------------------------------------------------------------ supervisor hand-over


def test_supervisor_finish_can_return_the_specialist_answer_verbatim():
    team = {"writer": ("writes", build_simple_agent(fake_model("The long careful answer."), []))}
    boss = fake_model(structured=[{"next": "writer", "instruction": "write it"},
                                  {"next": "FINISH", "instruction": ""}])
    graph = build_supervisor_graph(team, boss, checkpointer=InMemorySaver())
    out = run(graph.ainvoke(user("q"), cfg("v"), context=CTX))
    assert out["messages"][-1].content == "The long careful answer."
