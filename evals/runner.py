"""
EVALS

Behavioural regression cases in cases.jsonl. Each case has:
    input             the user message
    expect_contains   substrings that must appear in the final answer (case-insensitive)
    expect_not_contains  substrings that must NOT appear
    fake_response     what the fake model says in tests (so CI runs with no key)

Two ways to run:
    tests/test_evals.py      fake model, every push, proves the harness and wiring
    python -m evals          real model from .env, prints pass/fail per case
                             (set LANGSMITH_API_KEY to also log each run as a trace)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from agent.runtime import Context

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"


@dataclass
class Case:
    id: str
    input: str
    expect_contains: list[str] = field(default_factory=list)
    expect_not_contains: list[str] = field(default_factory=list)
    fake_response: str = ""


@dataclass
class CaseResult:
    case: Case
    output: str
    passed: bool
    reasons: list[str]


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def check(case: Case, output: str) -> CaseResult:
    low = output.lower()
    reasons = [f"missing: {s!r}" for s in case.expect_contains if s.lower() not in low]
    reasons += [f"present: {s!r}" for s in case.expect_not_contains if s.lower() in low]
    return CaseResult(case, output, not reasons, reasons)


async def run_case(graph, case: Case) -> CaseResult:
    config = {"configurable": {"thread_id": f"eval-{case.id}"}}
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": case.input}]},
        config=config,
        context=Context(user_id="eval"),
    )
    return check(case, str(result["messages"][-1].content))


async def run_all(graph_factory, cases: list[Case]) -> list[CaseResult]:
    """graph_factory(case) -> compiled graph. A fresh graph per case keeps them isolated."""
    results = []
    for case in cases:
        graph = await graph_factory(case)
        results.append(await run_case(graph, case))
    return results


def fresh_checkpointer() -> InMemorySaver:
    return InMemorySaver()
