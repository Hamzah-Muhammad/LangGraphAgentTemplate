"""
STRUCTURED OUTPUT SCHEMAS

Pydantic models the agent must fill in instead of writing prose.
Used by the planner graph (Plan) and available for `response_format=` on create_agent
when a caller needs a typed result (Answer).
"""

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Break a request into independent sub-tasks that can run in parallel."""

    steps: list[str] = Field(
        description="2-6 self-contained sub-tasks. Each must make sense on its own."
    )


class Answer(BaseModel):
    """Example typed final answer. Pass `response_format=ToolStrategy(Answer)` to use it."""

    summary: str = Field(description="One-paragraph answer.")
    confidence: float = Field(ge=0, le=1, description="0-1 how sure you are.")
    sources: list[str] = Field(default_factory=list, description="Tools or facts relied on.")
