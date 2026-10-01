# agent/shared/settings.py  |  shared helpers (not a block)
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


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return value.strip().lower() in {"1", "true", "yes", "on"} if value else default


@dataclass(frozen=True)
class Settings:
    context_window_tokens: int  # the model's real limit; compaction triggers below it
    todos: bool  # add the write_todos planning tool to the simple agent
    offload_chars: int  # tool results longer than this are saved to a file
    offload_dir: str  # where offloaded tool results live
    max_workers: int  # planner fan-out cap per round
    max_concurrency: int  # planner workers calling the model at the same time
    max_plan_rounds: int  # planner plan/review rounds per request
    max_eval_rounds: int  # evaluator-optimizer revision cap before human escalation
    max_supervisor_hops: int  # supervisor delegation cap
    request_timeout_s: float  # one model HTTP request
    tool_timeout_s: float  # one tool call (subagent tools are exempt)
    memory_inject: bool  # put the user's saved facts into the system prompt
    memory_inject_limit: int  # how many saved facts at most
    memory_fact_max_chars: int  # `remember` rejects a longer fact
    memory_require_approval: bool  # gate `remember` and `forget` behind human approval
    max_model_calls: int  # hard cap on model calls in one run of one agent


def get_settings() -> Settings:
    return Settings(
        context_window_tokens=_int("CONTEXT_WINDOW_TOKENS", 32_000),
        todos=_bool("AGENT_TODOS", False),
        offload_chars=_int("OFFLOAD_CHARS", 20_000),
        offload_dir=os.getenv("OFFLOAD_DIR", ".agent_files"),
        max_workers=_int("MAX_WORKERS", 6),
        max_concurrency=_int("MAX_CONCURRENCY", 3),
        max_plan_rounds=_int("MAX_PLAN_ROUNDS", 2),
        max_eval_rounds=_int("MAX_EVAL_ROUNDS", 3),
        max_supervisor_hops=_int("MAX_SUPERVISOR_HOPS", 6),
        request_timeout_s=_float("REQUEST_TIMEOUT_S", 120),
        tool_timeout_s=_float("TOOL_TIMEOUT_S", 60),
        memory_inject=_bool("MEMORY_INJECT", True),
        memory_inject_limit=_int("MEMORY_INJECT_LIMIT", 10),
        memory_fact_max_chars=_int("MEMORY_FACT_MAX_CHARS", 500),
        memory_require_approval=_bool("MEMORY_REQUIRE_APPROVAL", False),
        max_model_calls=_int("MAX_MODEL_CALLS", 25),
    )
