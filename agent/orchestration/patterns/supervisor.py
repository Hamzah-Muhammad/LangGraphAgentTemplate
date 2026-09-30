# agent/orchestration/patterns/supervisor.py  |  BLOCK 6 ORCHESTRATION
"""
SUPERVISOR (multi-agent)

    supervisor -> specialist A -> supervisor -> specialist B -> supervisor -> FINISH

A coordinator that never does the work. Each turn it picks ONE specialist and writes a
self-contained instruction, or finishes with the final answer. Control always returns to
the supervisor (Command(goto="supervisor")), so every routing decision is in one place
and on the audit trail.

Why supervisor and not swarm: supervisor is the 2026 default (Claude Code subagents,
OpenAI Agents SDK handoffs and LangGraph all converge on it). It costs about 2x the
model calls of a swarm, and buys more accurate routing and a single audit trail. Use a
swarm (agents hand off to each other directly) only when the flow is unpredictable and
latency matters more than auditability.

Specialists get only the instruction, not the whole conversation (private context),
and return a distilled result. Delegations are capped (MAX_SUPERVISOR_HOPS).
"""

from typing import Annotated, Any, Literal, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import Field, create_model
from typing_extensions import TypedDict

from agent.shared.runtime import Context

FINISH = "FINISH"

SUPERVISOR_PROMPT = """You coordinate a team of specialists. You never do the work yourself.

Team:
{team}

Read the conversation so far. Then either:
- pick ONE specialist for the next step and write it a self-contained instruction, or
- pick FINISH and put the complete final answer for the user in `instruction`.
Finish as soon as the request is fully answered. Do not repeat a step that already worked."""


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    instruction: NotRequired[str]
    hops: NotRequired[int]


def build_supervisor_graph(
    team: dict[str, tuple[str, Any]],
    model: BaseChatModel,
    *,
    max_hops: int = 6,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """team: {name: (description, compiled agent that takes {"messages": [...]})}."""
    names = tuple(team)
    decision_schema = create_model(
        "SupervisorDecision",
        next=(Literal[names + (FINISH,)], Field(description="Specialist to call, or FINISH.")),  # type: ignore[valid-type]
        instruction=(str, Field(description="Instruction for the specialist, or final answer.")),
    )
    decide = model.with_structured_output(decision_schema)
    roster = "\n".join(f"- {name}: {desc}" for name, (desc, _) in team.items())

    async def supervisor(state: SupervisorState) -> Command:
        hops = state.get("hops", 0)
        if hops >= max_hops:
            last = str(state["messages"][-1].content)
            note = f"Stopped after {max_hops} delegations. Latest result:\n\n{last}"
            return Command(goto=END, update={"messages": [AIMessage(note, name="supervisor")]})

        d = await decide.ainvoke(
            [SystemMessage(SUPERVISOR_PROMPT.format(team=roster)), *state["messages"]]
        )
        if d.next == FINISH:
            answer = AIMessage(content=d.instruction, name="supervisor")
            return Command(goto=END, update={"messages": [answer]})
        return Command(goto=d.next, update={"instruction": d.instruction, "hops": hops + 1})

    def specialist(name: str, agent):
        async def run(state: SupervisorState, runtime: Runtime[Context]) -> Command:
            out = await agent.ainvoke(
                {"messages": [{"role": "user", "content": state["instruction"]}]},
                context=runtime.context,
            )
            result = AIMessage(content=str(out["messages"][-1].content), name=name)
            return Command(goto="supervisor", update={"messages": [result]})

        return run

    def start(state: SupervisorState) -> dict:
        # The hop cap is per request. Without this reset, a thread that once hit the cap
        # stops every later request immediately.
        return {"hops": 0, "instruction": ""}

    builder = StateGraph(SupervisorState, context_schema=Context)
    builder.add_node("start", start)
    builder.add_node("supervisor", supervisor, destinations=names + (END,))
    for name, (_, agent) in team.items():
        builder.add_node(name, specialist(name, agent), destinations=("supervisor",))
    builder.add_edge(START, "start")
    builder.add_edge("start", "supervisor")
    return builder.compile(checkpointer=checkpointer, store=store, name="supervisor")
