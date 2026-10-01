# agent/orchestration/patterns/planner.py  |  BLOCK 6 ORCHESTRATION
"""
ORCHESTRATOR-WORKERS

    plan -> workers in parallel -> review -> (gaps and rounds left) more workers -> review
                                          -> (done) synthesize

Use for breadth-first requests: several independent directions (compare 4 vendors,
research 3 angles). Anthropic's Research system measured large gains on exactly these,
at roughly 15x the tokens of one agent. For one focused question it is pure overhead, so
the planner returns a single step, and the router sends such requests to the simple
agent anyway.

Rules from the research, all enforced here:
  - the orchestrator sees the recent conversation, so follow-ups plan correctly, and it
    writes every step self-contained because workers see only their own step
  - the plan is structured output you can log; a malformed plan falls back to one step
  - worker count per round is capped in code (MAX_WORKERS)
  - at most MAX_CONCURRENCY workers call the model at once (free hosts rate-limit)
  - after each round the orchestrator REVIEWS the results and may send workers after
    the gaps it finds, up to MAX_PLAN_ROUNDS rounds; a malformed review means "done"
  - workers return a distilled result with a word budget
  - the worker node retries once on a transient failure
  - per-request state resets at the start of every turn
"""

import asyncio
from typing import Annotated

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy, Send
from typing_extensions import TypedDict

from agent.context.history import latest_request, recent_conversation
from agent.model.structured import structured
from agent.shared.runtime import Context
from agent.shared.schemas import Plan, Review
from agent.shared.settings import get_settings

WORKER_WORD_BUDGET = 300

PLANNER_PROMPT = """Break the latest request into independent sub-tasks that can run in parallel.

Rules:
- At most {max_workers} steps. Fewer is better.
- Only split when the request has several independent directions.
  If it is one focused question, return exactly one step.
- Each step is done by someone who sees ONLY that step: no conversation, no other steps.
  Resolve every reference ("the third option", "same as before") into explicit words.

Example
Request: Compare Postgres, MySQL and SQLite for a small SaaS.
Steps:
- Postgres for a small SaaS: strengths, weaknesses, cost, operations burden
- MySQL for a small SaaS: strengths, weaknesses, cost, operations burden
- SQLite for a small SaaS: strengths, weaknesses, cost, operations burden

Conversation so far:
{history}

Latest request:
{request}"""

REVIEW_PROMPT = """You planned sub-tasks for a request and workers have reported back.

Request:
{request}

Results so far:
{results}

Is there enough to answer the request well? If yes, set done=true.
If something important is missing, set done=false and list at most {max_workers} new,
self-contained steps that fill exactly those gaps. Never repeat a step already done."""

WORKER_SUFFIX = "\n\nReturn a distilled result under {words} words. No preamble."

SYNTHESIZE_PROMPT = """Conversation so far:
{history}

Latest request:
{request}

Worker results:
{results}

Write the final answer for the user. Lead with the answer. Plain words, no preamble."""


def add_or_reset(current: list[str] | None, update: list[str] | None) -> list[str]:
    """Workers append in parallel; the plan node sends None to clear the previous turn."""
    if update is None:
        return []
    return (current or []) + update


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]  # steps for the round about to run
    results: Annotated[list[str], add_or_reset]  # workers append; plan resets per turn
    rounds: int  # plan/review rounds used this turn


class WorkerInput(TypedDict):
    task: str


def build_planner_graph(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    fallback_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    from agent.orchestration.graph import (
        build_simple_agent,  # local import: graph.py imports this module
    )

    settings = get_settings()
    max_workers, max_rounds = settings.max_workers, settings.max_plan_rounds
    # Workers are the simple agent, so they keep tools, memory, approval and guards.
    worker_agent = build_simple_agent(model, tools, fallback_model=fallback_model, name="worker")
    planner = structured(model, Plan)
    reviewer = structured(model, Review)
    semaphores: dict[int, asyncio.Semaphore] = {}  # one per event loop

    def gate() -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        return semaphores.setdefault(loop_id, asyncio.Semaphore(settings.max_concurrency))

    def clean(steps: list[str]) -> list[str]:
        return [s for s in steps if s.strip()][:max_workers]

    async def plan(state: PlannerState) -> dict:
        request = latest_request(state["messages"])
        prompt = PLANNER_PROMPT.format(
            max_workers=max_workers,
            history=recent_conversation(state["messages"]) or "(none)",
            request=request,
        )
        try:
            steps = clean((await planner.ainvoke([HumanMessage(prompt)])).steps)
        except Exception:  # malformed plan: do the whole request as one step
            steps = []
        return {"plan": steps or [request], "results": None, "rounds": 1}

    def fan_out(state: PlannerState) -> list[Send]:
        return [Send("worker", {"task": step}) for step in state["plan"]]

    async def worker(state: WorkerInput, runtime: Runtime[Context]) -> dict:
        task = state["task"] + WORKER_SUFFIX.format(words=WORKER_WORD_BUDGET)
        async with gate():
            out = await worker_agent.ainvoke(
                {"messages": [{"role": "user", "content": task}]},
                context=runtime.context,  # nested agents do not inherit context by themselves
            )
        return {"results": [f"### {state['task']}\n{out['messages'][-1].content}"]}

    async def review(state: PlannerState) -> dict:
        if state["rounds"] >= max_rounds:
            return {"plan": []}
        prompt = REVIEW_PROMPT.format(
            request=latest_request(state["messages"]),
            results="\n\n".join(state["results"]),
            max_workers=max_workers,
        )
        try:
            verdict: Review = await reviewer.ainvoke([HumanMessage(prompt)])
            more = [] if verdict.done else clean(verdict.missing_steps)
        except Exception:  # malformed review: treat as done rather than loop blindly
            more = []
        return {"plan": more, "rounds": state["rounds"] + (1 if more else 0)}

    def after_review(state: PlannerState) -> list[Send] | str:
        return fan_out(state) if state["plan"] else "synthesize"

    async def synthesize(state: PlannerState) -> dict:
        prompt = SYNTHESIZE_PROMPT.format(
            history=recent_conversation(state["messages"]) or "(none)",
            request=latest_request(state["messages"]),
            results="\n\n".join(state["results"]),
        )
        reply = await model.ainvoke([HumanMessage(prompt)])
        return {"messages": [AIMessage(content=reply.content)]}

    builder = StateGraph(PlannerState, context_schema=Context)
    builder.add_node("plan", plan)
    builder.add_node("worker", worker, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_node("review", review)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out, ["worker"])
    builder.add_edge("worker", "review")
    builder.add_conditional_edges("review", after_review, ["worker", "synthesize"])
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer, store=store, name="planner")
