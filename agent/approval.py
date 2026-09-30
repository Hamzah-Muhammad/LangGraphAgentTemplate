"""
HUMAN-IN-THE-LOOP GATE (part of Orchestration, kept separate so it is easy to find)

Any tool named in INTERRUPT_ON pauses the graph BEFORE the tool runs. The checkpointer
saves the state, run.py shows you the proposed call, and you approve / edit / reject.
The graph then resumes from the saved checkpoint on the same thread_id.

This is enforcement in code, not a prompt instruction. The model cannot skip it.
"""

from langchain.agents.middleware import HumanInTheLoopMiddleware

# tool name -> True (all three decisions allowed) or a dict of allowed decisions.
# PLACEHOLDER: gates the example tool so you can see the pause happen.
INTERRUPT_ON = {
    "example_tool": True,
}


def build_approval_middleware() -> HumanInTheLoopMiddleware:
    return HumanInTheLoopMiddleware(
        interrupt_on=INTERRUPT_ON,
        description_prefix="Tool call needs your approval",
    )
