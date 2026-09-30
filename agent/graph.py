"""
BLOCK 6: ORCHESTRATION

The loop. Zero model reasoning lives here; this file only decides what runs next.

Five modes, same six blocks underneath. Pick with `run.py --mode` or a graph name in
langgraph.json. docs/PATTERNS.md says when to use which.

  simple      create_agent tool loop. The default; covers most agents.
  planner     orchestrator-workers: plan, parallel workers, synthesize.
  router      rules then model classify, send to simple or planner.
  evaluator   generator + separate grader, capped rounds, human escalation.
  supervisor  coordinator delegating to specialist agents.

build_simple_agent is the one place middleware is assembled. Every pattern reuses it,
so guards, context handling, offload and approval apply everywhere.
"""

from pathlib import Path
from typing import Literal, get_args

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from agent.approval import build_approval_middleware
from agent.context import build_context_middleware
from agent.mcp import load_mcp_tools
from agent.offload import OffloadLargeToolResults
from agent.reliability import build_reliability_middleware
from agent.runtime import Context
from agent.settings import get_settings
from agent.skills import skills_index
from agents import SUBAGENT_BUILDERS
from tools import ALL_TOOLS

Mode = Literal["simple", "planner", "router", "evaluator", "supervisor"]
MODES: tuple[str, ...] = get_args(Mode)
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    """BLOCK 3: SYSTEM PROMPT. Markdown file plus the one-line skills index."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    index = skills_index()
    return f"{prompt}\n\n{index}" if index else prompt


async def gather_tools(model: BaseChatModel, include_mcp: bool = True) -> list[BaseTool]:
    """Python tools + subagent tools + MCP tools, one list."""
    tools = [*ALL_TOOLS, *(build(model) for build in SUBAGENT_BUILDERS)]
    if include_mcp:
        tools += await load_mcp_tools()
    return tools


def build_simple_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    system_prompt: str | None = None,
    fallback_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | bool | None = None,
    store: BaseStore | None = None,
    name: str = "agent",
):
    settings = get_settings()
    # Order is load-bearing: first = outermost wrapper. See agent/reliability.py.
    middleware = [
        *build_reliability_middleware(fallback_model),  # guards, retries, fallback
        *build_context_middleware(model, settings),  # Context Window
        OffloadLargeToolResults(settings.offload_chars, settings.offload_dir),
        build_approval_middleware(),  # human gate
    ]
    return create_agent(
        model=model,  # Model
        tools=tools,  # Tools
        system_prompt=system_prompt or load_system_prompt(),  # System Prompt
        middleware=middleware,
        context_schema=Context,
        checkpointer=checkpointer,  # Memory (short-term)
        store=store,  # Memory (long-term)
        name=name,
        # response_format=ToolStrategy(Answer),   # uncomment for typed final answers
    )


async def build_graph(
    model: BaseChatModel,
    *,
    mode: Mode = "simple",
    fallback_model: BaseChatModel | None = None,
    grader_model: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    include_mcp: bool = True,
):
    """Assemble the blocks into one compiled graph for the chosen mode."""
    # Local imports: the pattern modules import build_simple_agent from here.
    from agent.patterns.evaluator import build_evaluator_graph
    from agent.patterns.planner import build_planner_graph
    from agent.patterns.router import build_router_graph
    from agent.patterns.supervisor import build_supervisor_graph
    from agents.team import build_team

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; pick one of {MODES}")
    settings = get_settings()
    tools = await gather_tools(model, include_mcp=include_mcp)
    persist = {"checkpointer": checkpointer, "store": store}

    if mode == "simple":
        return build_simple_agent(model, tools, fallback_model=fallback_model, **persist)
    if mode == "planner":
        return build_planner_graph(model, tools, fallback_model=fallback_model, **persist)
    if mode == "evaluator":
        return build_evaluator_graph(
            model, grader_model or model, max_rounds=settings.max_eval_rounds, **persist
        )
    if mode == "supervisor":
        team = build_team(model, tools, fallback_model=fallback_model)
        return build_supervisor_graph(
            team, model, max_hops=settings.max_supervisor_hops, **persist
        )
    # router: nested graphs inherit the router's checkpointer, so they get none of their own
    routes = {
        "simple": (
            "One focused question, a lookup, or a quick task.",
            build_simple_agent(model, tools, fallback_model=fallback_model),
        ),
        "planner": (
            "Several independent parts to research or compare in parallel.",
            build_planner_graph(model, tools, fallback_model=fallback_model),
        ),
    }
    return build_router_graph(routes, model, default="simple", **persist)
