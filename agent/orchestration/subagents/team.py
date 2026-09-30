# agent/orchestration/subagents/team.py  |  BLOCK 6 ORCHESTRATION
"""
Placeholder team for the supervisor pattern. Each member is the simple agent with its
own prompt and tool subset. Edit descriptions carefully: they are all the supervisor
knows about who does what.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

RESEARCHER_PROMPT = """You are the researcher. Use tools to find facts for the instruction
you are given. Report the facts and which tool gave each one. No opinions, no prose polish.
Keep it under 300 words."""

WRITER_PROMPT = """You are the writer. Turn the notes in the instruction into a clear,
well-structured answer for the user. Do not add facts that are not in the notes.
Lead with the answer. Plain words."""


def build_team(
    model: BaseChatModel,
    tools: list[BaseTool],
    *,
    fallback_model: BaseChatModel | None = None,
) -> dict:
    from agent.orchestration.graph import build_simple_agent

    return {
        "researcher": (
            "Finds facts with tools and reports them with their source.",
            build_simple_agent(
                model, tools, system_prompt=RESEARCHER_PROMPT,
                fallback_model=fallback_model, name="researcher",
            ),
        ),
        "writer": (
            "Turns notes into the final user-facing answer. Uses no tools.",
            build_simple_agent(
                model, [], system_prompt=WRITER_PROMPT,
                fallback_model=fallback_model, name="writer",
            ),
        ),
    }
