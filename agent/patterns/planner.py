"""
ORCHESTRATOR-WORKERS

    plan (structured) -> fan out N workers in parallel (Send) -> synthesize

Use for breadth-first requests: several independent directions (compare 4 vendors,
research 3 angles). Anthropic's Research system measured large gains on exactly these,
at roughly 15x the tokens of one agent. For one focused question it is pure overhead,
so the planner is told to return a single step, and the router sends such requests to
the simple agent anyway.

Rules from the research, all enforced here:
  - the plan is structured output you can log and inspect
  - worker count is capped in code (MAX_WORKERS), not just asked for in the prompt
  - the planner gets an example of a good decomposition
  - workers return a distilled result with a word budget
  - the worker node has a retry policy for transient failures
"""

import operator
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

from agent.runtime import Context
from agent.schemas import Plan
from agent.settings import get_settings

WORKER_WORD_BUDGET = 300

PLANNER_PROMPT = """Break the request into independent sub-tasks that can run in parallel.

Rules:
- At most {max_workers} steps. Fewer is better.
- Only split when the request has several independent directions.
  If it is one focused question, return exactly one step: the request itself.
- Each step must make sense on its own, without seeing the others.

Example
Request: Compare Postgres, MySQL and SQLite for a small SaaS.
Steps:
- Postgres for a small SaaS: strengths, weaknesses, cost, operations burden
- MySQL for a small SaaS: strengths, weaknesses, cost, operations burden
- SQLite for a small SaaS: strengths, weaknesses, cost, operations burden

Request:
{request}"""

WORKER_SUFFIX = "\n\nReturn a distilled result under {words} words. No preamble."

SYNTHESIZE_PROMPT = """Original request:
{request}

Worker results:
{results}

Write the final answer for the user. Lead with the answer. Plain words, no preamble."""


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: list[str]
    results: Annotated[list[str], operator.add]  # each worker appends one entry


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
    from agent.graph import build_simple_agent  # local import: graph.py imports this module

    max_workers = get_settings().max_workers
    # Workers are the simple agent, so they keep tools, memory, approval and guards.
    worker_agent = build_simple_agent(model, tools, fallback_model=fallback_model, name="worker")
    planner_model = model.with_structured_output(Plan)

    async def plan(state: PlannerState) -> dict:
        request = str(state["messages"][-1].content)
        prompt = PLANNER_PROMPT.format(max_workers=max_workers, request=request)
        result: Plan = await planner_model.ainvoke([HumanMessage(content=prompt)])
        # Empty plan would skip workers AND synthesize; cap enforced in code.
        steps = [s for s in result.steps if s.strip()][:max_workers] or [request]
        return {"plan": steps, "results": []}

    def fan_out(state: PlannerState) -> list[Send]:
        return [Send("worker", {"task": step}) for step in state["plan"]]

    async def worker(state: WorkerInput, runtime: Runtime[Context]) -> dict:
        task = state["task"] + WORKER_SUFFIX.format(words=WORKER_WORD_BUDGET)
        out = await worker_agent.ainvoke(
            {"messages": [{"role": "user", "content": task}]},
            context=runtime.context,  # nested agents do not inherit context by themselves
        )
        return {"results": [f"### {state['task']}\n{out['messages'][-1].content}"]}

    async def synthesize(state: PlannerState) -> dict:
        prompt = SYNTHESIZE_PROMPT.format(
            request=state["messages"][-1].content, results="\n\n".join(state["results"])
        )
        reply = await model.ainvoke([HumanMessage(content=prompt)])
        return {"messages": [AIMessage(content=reply.content)]}

    builder = StateGraph(PlannerState, context_schema=Context)
    builder.add_node("plan", plan)
    builder.add_node("worker", worker, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out, ["worker"])
    builder.add_edge("worker", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer, store=store, name="planner")
