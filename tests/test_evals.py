"""Runs evals/cases.jsonl through the fake model, and unit-tests the trajectory checks."""

import asyncio

from langchain_core.messages import AIMessage

from agent.orchestration.graph import build_graph
from evals.runner import Case, check, fresh_checkpointer, load_cases, run_all, tool_trajectory
from tests.fakes import fake_model


def test_all_eval_cases_pass_with_fake_model():
    cases = load_cases()
    assert cases, "no eval cases found"

    async def factory(case):
        script = case.fake_script or [case.fake_response]
        return await build_graph(
            fake_model(*script), checkpointer=fresh_checkpointer(), include_mcp=False
        )

    results = asyncio.run(run_all(factory, cases))
    failed = [(r.case.id, r.reasons) for r in results if not r.passed]
    assert not failed, failed


def test_trajectory_checks_catch_wrong_paths():
    case = Case(id="t", input="x", expect_tools=["search", "write"], forbid_tools=["delete"],
                max_tool_calls=3)
    assert check(case, "ok", ["search", "read", "write"]).passed
    assert not check(case, "ok", ["write", "search"]).passed  # wrong order
    assert not check(case, "ok", ["search", "delete", "write"]).passed  # forbidden tool
    assert not check(case, "ok", ["search", "x", "x", "write"]).passed  # too many calls


def test_tool_trajectory_reads_ai_tool_calls_in_order():
    msgs = [
        AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "b", "args": {}, "id": "2"}]),
        AIMessage(content="done"),
    ]
    assert tool_trajectory(msgs) == ["a", "b"]
