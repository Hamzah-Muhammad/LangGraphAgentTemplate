"""
SUBAGENTS

Specialist agents wrapped as tools for the main agent. Each subagent is a full
create_agent with its own prompt and tool subset. The main agent delegates by calling
the tool; it never sees the subagent's internal loop.

Register builders here. Each takes the shared model and returns a tool.
"""

from agents.specialist import build_specialist_tool

SUBAGENT_BUILDERS = [
    build_specialist_tool,
]
