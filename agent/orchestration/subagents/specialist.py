# agent/orchestration/subagents/specialist.py  |  BLOCK 6 ORCHESTRATION
"""
Placeholder specialist subagent. Copy this file for each real specialist
(researcher, coder, reviewer, ...).

Pattern: build an inner create_agent with a narrow prompt and a narrow tool list, then
wrap `invoke` in a @tool so the outer agent can delegate. The docstring on the tool is
what tells the outer model WHEN to delegate; make it specific.

Stateless: each call starts fresh. For a subagent that must remember across calls on the
same thread, compile with `checkpointer=True` and add ToolCallLimitMiddleware(run_limit=1)
on the outer agent to prevent parallel calls into the same checkpoint namespace.
"""

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from agent.tools.example_tool import example_tool

SPECIALIST_PROMPT = """You are a specialist. You get one narrow task and return one
concise result. Use your tools if needed. Do not ask questions; make reasonable
assumptions and state them."""


def build_specialist_tool(model: BaseChatModel) -> BaseTool:
    from agent.orchestration.graph import build_simple_agent  # local: graph.py imports this

    # Built with build_simple_agent, NOT bare create_agent, so the subagent keeps every
    # guard: call cap, retries, context handling, offload and the approval gate. A bare
    # create_agent here let the main agent run gated tools unapproved by delegating them.
    specialist = build_simple_agent(
        model, [example_tool], system_prompt=SPECIALIST_PROMPT, name="specialist"
    )

    @tool
    async def ask_specialist(task: str) -> str:
        """Delegate a self-contained sub-task to the specialist agent and get its result.
        Use when a task needs focused, multi-step work that would clutter the main thread."""
        result = await specialist.ainvoke({"messages": [{"role": "user", "content": task}]})
        return result["messages"][-1].content

    # A whole agent runs inside this tool and it may wait for a human approval,
    # so the per-tool timeout must not cancel it.
    ask_specialist.metadata = {"long_running": True}
    return ask_specialist
