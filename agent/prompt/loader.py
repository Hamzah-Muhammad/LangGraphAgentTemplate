# agent/prompt/loader.py  |  BLOCK 3 SYSTEM PROMPT
"""Builds the system prompt: system.md plus the one-line skills index."""

from pathlib import Path

from agent.prompt.skills import skills_index

PROMPT_PATH = Path(__file__).resolve().parent / "system.md"


def load_system_prompt() -> str:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    index = skills_index()
    return f"{prompt}\n\n{index}" if index else prompt
