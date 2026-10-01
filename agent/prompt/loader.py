# agent/prompt/loader.py  |  BLOCK 3 SYSTEM PROMPT
"""Builds the system prompt: system.md plus the one-line skills index."""

from pathlib import Path

from agent.prompt.skills import skills_index

PROMPT_PATH = Path(__file__).resolve().parent / "system.md"


def load_system_prompt(with_skills: bool = True) -> str:
    """with_skills=False for model calls that have no tools (they cannot call load_skill)."""
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    index = skills_index() if with_skills else ""
    return f"{prompt}\n\n{index}" if index else prompt
