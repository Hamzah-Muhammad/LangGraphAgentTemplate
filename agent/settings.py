"""
SETTINGS

Every tunable number in one place, read from the environment at call time so tests can
override them. Defaults are safe for a 32k-context open model.
"""

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return value.strip().lower() in {"1", "true", "yes", "on"} if value else default


@dataclass(frozen=True)
class Settings:
    context_window_tokens: int  # the model's real limit; compaction triggers below it
    todos: bool  # add the write_todos planning tool to the simple agent
    offload_chars: int  # tool results longer than this are saved to a file
    offload_dir: str  # where offloaded tool results live
    max_workers: int  # planner fan-out cap
    max_eval_rounds: int  # evaluator-optimizer revision cap before human escalation
    max_supervisor_hops: int  # supervisor delegation cap


def get_settings() -> Settings:
    return Settings(
        context_window_tokens=_int("CONTEXT_WINDOW_TOKENS", 32_000),
        todos=_bool("AGENT_TODOS", False),
        offload_chars=_int("OFFLOAD_CHARS", 20_000),
        offload_dir=os.getenv("OFFLOAD_DIR", ".agent_files"),
        max_workers=_int("MAX_WORKERS", 6),
        max_eval_rounds=_int("MAX_EVAL_ROUNDS", 3),
        max_supervisor_hops=_int("MAX_SUPERVISOR_HOPS", 6),
    )
