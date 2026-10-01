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
    judge                a rubric in plain words. A judge model (the grader model, else the
                         primary) answers PASS or FAIL against it. For what substrings cannot
                         check: tone, correctness of an explanation. Needs a judge model, so
                         the fake-model tests skip it; `python -m evals` runs it.
    mode                 workflow mode to run the case in (default "simple")
    fake_structured      scripted structured-output answers for tests (route, plan...)
    fake_response        what the fake model answers in tests (no tool calls), or
    fake_script          a list of scripted model turns for tests, e.g.
                         [{"tool_calls": [{"name": "x", "args": {...}}]}, "final text"]

Why trajectory: a right answer reached the wrong way (skipped a lookup, looped five
times, called a write tool) is a latent bug. 2026 evals check the path, not just the end.

The tool path is recorded by a callback (ToolTrace), not read off the final messages,
so it also sees tools called by nested agents: planner workers, supervisor specialists,
the router's chosen route. A case can set "mode" to evaluate any workflow mode.

Gated tools are auto-approved during evals so a run never blocks on a human.

    tests/test_evals.py   fake model, every push: proves harness and wiring
    python -m evals       real model from .env: prints pass/fail per case
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler
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
    judge: str = ""
    mode: str = "simple"
    fake_structured: list | dict | None = None
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


class ToolTrace(BaseCallbackHandler):
    """Records every tool call in a run, in start order, nested agents included."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name")
        if name:
            self.names.append(name)


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


JUDGE_PROMPT = (
    "You grade an AI answer against a rubric. Be strict. Reply with exactly PASS or FAIL "
    "on the first line, then one short reason on the second."
)


async def judge_answer(model, case: Case, output: str) -> tuple[bool, str]:
    """Ask a judge model. Anything that is not a clear PASS counts as FAIL."""
    reply = await model.ainvoke([
        ("system", JUDGE_PROMPT),
        ("human", f"Request:\n{case.input}\n\nRubric:\n{case.judge}\n\nAnswer:\n{output}"),
    ])
    text = str(reply.content).strip()
    first = text.splitlines()[0].strip().upper() if text else ""
    return first.startswith("PASS"), text


def _approve_all(interrupts) -> Command:
    resume = {}
    for i in interrupts:
        n = len(i.value.get("action_requests", [])) if isinstance(i.value, dict) else 0
        resume[i.id] = {"decisions": [{"type": "approve"}] * max(n, 1)}
    return Command(resume=resume)


async def run_case(graph, case: Case, judge_model=None) -> CaseResult:
    trace = ToolTrace()
    config = {"configurable": {"thread_id": f"eval-{case.id}"}, "callbacks": [trace]}
    context = Context(user_id="eval")
    payload = {"messages": [{"role": "user", "content": case.input}]}
    for _ in range(MAX_AUTO_APPROVALS):
        result = await graph.ainvoke(payload, config=config, context=context)
        if not result.get("__interrupt__"):
            break
        payload = _approve_all(result["__interrupt__"])
    output = str(result["messages"][-1].content)
    outcome = check(case, output, trace.names)
    if case.judge and judge_model is not None:
        ok, verdict = await judge_answer(judge_model, case, output)
        if not ok:
            outcome.reasons.append(f"judge: {verdict[:200]}")
            outcome.passed = False
    return outcome


async def run_all(graph_factory, cases: list[Case], judge_model=None) -> list[CaseResult]:
    """graph_factory(case) -> compiled graph. A fresh graph per case keeps them isolated."""
    results = []
    for case in cases:
        graph = await graph_factory(case)
        results.append(await run_case(graph, case, judge_model))
    return results


def fresh_checkpointer() -> InMemorySaver:
    return InMemorySaver()
