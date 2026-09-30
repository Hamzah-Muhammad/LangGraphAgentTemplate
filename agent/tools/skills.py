# agent/tools/skills.py  |  BLOCK 2 TOOLS
"""load_skill: pull a full SKILL.md procedure into context on demand.

The skills themselves and the index live in the prompt block: agent/prompt/skills.py.
"""

from langchain.tools import tool

from agent.prompt.skills import discover_skills, skill_body


@tool
def load_skill(name: str) -> str:
    """Load the full instructions for a named skill from the Skills index.
    Call this before starting a task that a listed skill covers, then follow it."""
    body = skill_body(name)
    if body is None:
        available = ", ".join(discover_skills()) or "none"
        return f"error: unknown skill {name!r}. Available: {available}"
    return body
