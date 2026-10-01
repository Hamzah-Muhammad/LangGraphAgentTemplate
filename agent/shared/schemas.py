# agent/shared/schemas.py  |  shared helpers (not a block)
"""
STRUCTURED OUTPUT SCHEMAS

Pydantic models the model must fill in instead of writing prose. Every decision a
workflow makes from model output goes through one of these, so it can be logged,
validated and tested. Route and supervisor schemas are built at runtime from the route
names (see agent/orchestration/patterns/router.py and supervisor.py).
"""

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Break a request into independent sub-tasks that can run in parallel."""

    steps: list[str] = Field(
        description="Self-contained sub-tasks. One step if the request is a single focused ask."
    )


class Review(BaseModel):
    """The orchestrator's check of worker results after a round."""

    done: bool = Field(description="True if the results are enough to answer the request.")
    missing_steps: list[str] = Field(
        default_factory=list, description="New self-contained steps that fill the gaps."
    )


class Verdict(BaseModel):
    """A grader's decision on a draft."""

    passed: bool = Field(description="True only if the draft fully meets the request.")
    feedback: str = Field(description="What is missing or wrong, and how to fix it.")


class Answer(BaseModel):
    """Example typed final answer. Pass `response_format=ToolStrategy(Answer)` to use it."""

    summary: str = Field(description="One-paragraph answer.")
    confidence: float = Field(ge=0, le=1, description="0-1 how sure you are.")
    sources: list[str] = Field(default_factory=list, description="Tools or facts relied on.")
