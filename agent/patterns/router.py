"""
ROUTING

    classify -> exactly one route -> END

Classify first, then send the request to the path best equipped for it. Rules run
before the model: every rule that matches is one model call saved and one failure
surface removed. The model classifier only sees what the rules could not decide, and
its answer is a Literal of the real route names, so it cannot invent a route. If the
classifier itself fails, the default route runs.

Default wiring (agent/graph.py): route between the simple agent and the planner.
"""

import re
from collections.abc import Sequence
from typing import Annotated, Any, Literal, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command
from pydantic import Field, create_model
from typing_extensions import TypedDict

from agent.runtime import Context

# (regex, route). First match wins. Case-insensitive.
DEFAULT_RULES: list[tuple[str, str]] = [
    (r"\b(compare|comparison|versus|vs\.?|pros and cons|research|survey|options for)\b",
     "planner"),
]

ROUTER_PROMPT = """Pick the route that best fits the user's request.

Routes:
{routes}"""


class RouterState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: NotRequired[str]
    routed_by: NotRequired[str]  # "rule" | "model" | "default", for logs and evals


def build_router_graph(
    routes: dict[str, tuple[str, Any]],
    model: BaseChatModel,
    *,
    rules: Sequence[tuple[str, str]] = DEFAULT_RULES,
    default: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """routes: {name: (description, compiled graph that takes {"messages": [...]})}."""
    names = tuple(routes)
    default = default or names[0]
    route_choice = create_model(
        "RouteChoice",
        route=(Literal[names], Field(description="The route name.")),  # type: ignore[valid-type]
    )
    classifier = model.with_structured_output(route_choice)
    menu = "\n".join(f"- {name}: {desc}" for name, (desc, _) in routes.items())

    def match_rules(text: str) -> str | None:
        for pattern, route in rules:
            if route in routes and re.search(pattern, text, re.IGNORECASE):
                return route
        return None

    async def classify(state: RouterState) -> Command:
        text = str(state["messages"][-1].content)
        route, how = match_rules(text), "rule"
        if route is None:
            try:
                choice = await classifier.ainvoke(
                    [SystemMessage(ROUTER_PROMPT.format(routes=menu)), HumanMessage(text)]
                )
                route, how = choice.route, "model"
            except Exception:  # a broken classifier must not break the request
                route, how = default, "default"
        return Command(goto=route, update={"route": route, "routed_by": how})

    def route_node(graph):
        async def run(state: RouterState, runtime: Runtime[Context]) -> dict:
            out = await graph.ainvoke({"messages": state["messages"]}, context=runtime.context)
            return {"messages": [out["messages"][-1]]}

        return run

    builder = StateGraph(RouterState, context_schema=Context)
    builder.add_node("classify", classify, destinations=names)
    for name, (_, graph) in routes.items():
        builder.add_node(name, route_node(graph))
        builder.add_edge(name, END)
    builder.add_edge(START, "classify")
    return builder.compile(checkpointer=checkpointer, store=store, name="router")
