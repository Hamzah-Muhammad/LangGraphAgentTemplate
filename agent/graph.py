"""
BLOCK 6: ORCHESTRATION

The loop. Zero model reasoning lives here; this file only decides what runs next.

Two graphs, same blocks:

  simple   create_agent: START -> [middleware] -> model <-> tools -> END
           The plug-and-play path. Covers 90% of agents.

  planner  Hand-written StateGraph for complex work:
               plan -> fan out N workers in parallel (Send) -> synthesize
           Each worker IS the simple agent, so it keeps tools, memory, approval.
           Copy this pattern when create_agent cannot express your flow.

`make_graph` at the bottom is what langgraph.json points at (LangGraph Studio / Platform).
"""

import operator
from pathlib import Path
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.types import Send
from typing_extensions import TypedDict

from agent.approval import build_approval_middleware
from agent.context import build_context_middleware
from agent.mcp import load_mcp_tools
from agent.reliability import build_reliability_middleware
from agent.runtime import Context
from agent.schemas import Plan
from agents import SUBAGENT_BUILDERS
from tools import ALL_TOOLS

Mode = Literal["simple", "planner"]
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    """BLOCK 3: SYSTEM PROMPT. Read from Markdown so behaviour is editable without code."""
    return PROMPT_PATH.read_text(encoding="utf-8")


async def gather_tools(model: BaseChatModel, include_mcp: bool = True) -> list[BaseTool]:
    """Python tools + subagent tools + MCP tools, one list."""
    tools = [*ALL_TOOLS, *(build(model) for build in SUBAGENT_BUILDERS)]
    if include_mcp:
        tools += await load_mcp_tools()
    return tools


# --------------------------------------------------------------------------- simple


def build_simple_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    fallback_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | bool | None = None,
    store: BaseStore | None = None,
    name: str = "agent",
):
    middleware = [
        *build_reliability_middleware(fallback_model),  # guards, retries, fallback
        *build_context_middleware(model),  # Context Window
        build_approval_middleware(),  # human gate
    ]
    return create_agent(
        model=model,  # Model
        tools=tools,  # Tools
        system_prompt=load_system_prompt(),  # System Prompt
        middleware=middleware,
        context_schema=Context,
        checkpointer=checkpointer,  # Memory (short-term)
        store=store,  # Memory (long-term)
        name=name,
        # response_format=ToolStrategy(Answer),   # uncomment for typed final answers
    )


# --------------------------------------------------------------------------- planner


class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # coerces dicts, dedupes by id
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
    # Workers are the simple agent compiled as a subgraph (inherits checkpointer/store).
    worker_agent = build_simple_agent(
        model, tools, fallback_model=fallback_model, name="worker"
    )
    planner_model = model.with_structured_output(Plan)

    async def plan(state: PlannerState) -> dict:
        request = state["messages"][-1].content
        result: Plan = await planner_model.ainvoke(
            [HumanMessage(content=f"Break this request into sub-tasks:\n\n{request}")]
        )
        return {"plan": result.steps, "results": []}

    def fan_out(state: PlannerState) -> list[Send]:
        # One worker per step, all in parallel. Send() creates a branch per item.
        return [Send("worker", {"task": step}) for step in state["plan"]]

    async def worker(state: WorkerInput) -> dict:
        out = await worker_agent.ainvoke(
            {"messages": [{"role": "user", "content": state["task"]}]}
        )
        return {"results": [f"### {state['task']}\n{out['messages'][-1].content}"]}

    async def synthesize(state: PlannerState) -> dict:
        request = state["messages"][-1].content
        joined = "\n\n".join(state["results"])
        reply = await model.ainvoke(
            [
                HumanMessage(
                    content=(
                        f"Original request:\n{request}\n\n"
                        f"Worker results:\n{joined}\n\n"
                        "Write the final answer for the user. Plain words, no preamble."
                    )
                )
            ]
        )
        return {"messages": [AIMessage(content=reply.content)]}

    builder = StateGraph(PlannerState, context_schema=Context)
    builder.add_node("plan", plan)
    builder.add_node("worker", worker)
    builder.add_node("synthesize", synthesize)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out, ["worker"])
    builder.add_edge("worker", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile(checkpointer=checkpointer, store=store, name="planner")


# --------------------------------------------------------------------------- entry


async def build_graph(
    model: BaseChatModel,
    *,
    mode: Mode = "simple",
    fallback_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    include_mcp: bool = True,
):
    """Assemble the six blocks into one compiled, runnable graph."""
    tools = await gather_tools(model, include_mcp=include_mcp)
    if mode == "planner":
        return build_planner_graph(
            model, tools, fallback_model=fallback_model, checkpointer=checkpointer, store=store
        )
    return build_simple_agent(
        model, tools, fallback_model=fallback_model, checkpointer=checkpointer, store=store
    )


async def make_graph(config: dict | None = None):
    """Factory for langgraph.json (LangGraph Studio / Platform supply persistence)."""
    from dotenv import load_dotenv

    from agent.model import build_fallback_model, build_model

    load_dotenv()
    mode = (config or {}).get("configurable", {}).get("mode", "simple")
    return await build_graph(build_model(), mode=mode, fallback_model=build_fallback_model())
