"""Runs evals/cases.jsonl through the fake model. Proves the eval harness on every push."""

import asyncio

from agent.graph import build_graph
from evals.runner import fresh_checkpointer, load_cases, run_all
from tests.fakes import fake_model


def test_all_eval_cases_pass_with_fake_model():
    cases = load_cases()
    assert cases, "no eval cases found"

    async def factory(case):
        return await build_graph(
            fake_model(case.fake_response), checkpointer=fresh_checkpointer(), include_mcp=False
        )

    results = asyncio.run(run_all(factory, cases))
    failed = [(r.case.id, r.reasons) for r in results if not r.passed]
    assert not failed, failed
