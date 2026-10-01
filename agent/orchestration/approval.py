# agent/orchestration/approval.py  |  BLOCK 6 ORCHESTRATION
"""
HUMAN-IN-THE-LOOP GATE (part of Orchestration, kept separate so it is easy to find)

Any tool named in INTERRUPT_ON pauses the graph BEFORE the tool runs. The checkpointer
saves the state, run.py shows you the proposed call, and you approve / edit / reject.
The graph then resumes from the saved checkpoint on the same thread_id.

This is enforcement in code, not a prompt instruction. The model cannot skip it.
"""

from langchain.agents.middleware import HumanInTheLoopMiddleware

from agent.shared.settings import get_settings

# tool name -> True (all three decisions allowed) or a dict of allowed decisions.
# PLACEHOLDER: gates the example tool so you can see the pause happen.
INTERRUPT_ON = {
    "example_tool": True,
}


def build_approval_middleware() -> HumanInTheLoopMiddleware:
    gated = dict(INTERRUPT_ON)
    if get_settings().memory_require_approval:
        # Saved facts reach the system prompt, so a poisoned `remember` is a prompt attack.
        gated |= {"remember": True, "forget": True}
    return HumanInTheLoopMiddleware(
        interrupt_on=gated,
        description_prefix="Tool call needs your approval",
    )
