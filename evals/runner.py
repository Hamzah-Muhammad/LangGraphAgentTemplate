"""
EVALS: outcome AND trajectory

Each line of cases.jsonl is one case:
    input                user message
    expect_contains      substrings the final answer must contain (case-insensitive)
    expect_not_contains  substrings it must not contain
    expect_tools         tool names that must be called, in this order (other calls may
                         come in between). This is the trajectory check.
    forbid_tools         tool names that must NOT be called
    max_tool_calls       upper bound on total tool calls (catches loops)
    fake_response        what the fake model answers in tests (no tool calls), or
    fake_script          a list of scripted model turns for tests, e.g.
                         [{"tool_calls": [{"name": "x", "args": {...}}]}, "final text"]

Why trajectory: a right answer reached the wrong way (skipped a lookup, looped five
times, called a write tool) is a latent bug. 2026 evals check the path, not just the end.

Gated tools are auto-approved during evals so a run never blocks on a human.

    tests/test_evals.py   fake model, every push: proves harness and wiring
    python -m evals       real model from .env: prints pass/fail per case
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.shared.runtime import Context

CASES_PATH = Path(__file__).resolve().parent / "cases.jsonl"
MAX_AUTO_APPROVALS = 10


@dataclass
class Case:
    id: str
    input: str
    expect_contains: list[str] = field(default_factory=list)
    expect_not_contains: list[str] = field(default_factory=list)
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    max_tool_calls: int | None = None
    fake_response: str = ""
    fake_script: list = field(default_factory=list)


@dataclass
class CaseResult:
    case: Case
    output: str
    tools: list[str]
    passed: bool
    reasons: list[str]


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def tool_trajectory(messages) -> list[str]:
    return [tc["name"] for m in messages if m.type == "ai" for tc in (m.tool_calls or [])]


def _is_subsequence(wanted: list[str], actual: list[str]) -> bool:
    it = iter(actual)
    return all(name in it for name in wanted)


def check(case: Case, output: str, tools: list[str]) -> CaseResult:
    low = output.lower()
    reasons = [f"missing: {s!r}" for s in case.expect_contains if s.lower() not in low]
    reasons += [f"present: {s!r}" for s in case.expect_not_contains if s.lower() in low]
    if not _is_subsequence(case.expect_tools, tools):
        reasons.append(f"trajectory: expected {case.expect_tools} in order, got {tools}")
    reasons += [f"forbidden tool called: {t}" for t in case.forbid_tools if t in tools]
    if case.max_tool_calls is not None and len(tools) > case.max_tool_calls:
        reasons.append(f"too many tool calls: {len(tools)} > {case.max_tool_calls}")
    return CaseResult(case, output, tools, not reasons, reasons)


def _approve_all(interrupts) -> Command:
    resume = {}
    for i in interrupts:
        n = len(i.value.get("action_requests", [])) if isinstance(i.value, dict) else 0
        resume[i.id] = {"decisions": [{"type": "approve"}] * max(n, 1)}
    return Command(resume=resume)


async def run_case(graph, case: Case) -> CaseResult:
    config = {"configurable": {"thread_id": f"eval-{case.id}"}}
    context = Context(user_id="eval")
    payload = {"messages": [{"role": "user", "content": case.input}]}
    for _ in range(MAX_AUTO_APPROVALS):
        result = await graph.ainvoke(payload, config=config, context=context)
        if not result.get("__interrupt__"):
            break
        payload = _approve_all(result["__interrupt__"])
    messages = result["messages"]
    return check(case, str(messages[-1].content), tool_trajectory(messages))


async def run_all(graph_factory, cases: list[Case]) -> list[CaseResult]:
    """graph_factory(case) -> compiled graph. A fresh graph per case keeps them isolated."""
    results = []
    for case in cases:
        graph = await graph_factory(case)
        results.append(await run_case(graph, case))
    return results


def fresh_checkpointer() -> InMemorySaver:
    return InMemorySaver()
