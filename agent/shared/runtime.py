# agent/shared/runtime.py  |  shared helpers (not a block)
"""
RUNTIME CONTEXT

Per-run, read-only data that is NOT part of the conversation: who the user is, feature
flags, tenant id. Tools and middleware read it through `runtime.context`.
It never enters the prompt unless a tool puts it there.
"""

from dataclasses import dataclass


@dataclass
class Context:
    user_id: str = "anonymous"
