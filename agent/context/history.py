# agent/context/history.py  |  BLOCK 4 CONTEXT WINDOW
"""
CONVERSATION CONTEXT for workflow steps that are not the chat agent itself

The planner, the router's classifier and the synthesizer each make one model call from
a short prompt. If that prompt holds only the latest message, a follow-up such as
"now do the same for the third option" is meaningless. recent_conversation() renders
the last few user and assistant turns as plain text, capped in size, so these steps can
resolve references without receiving the whole thread. Tool traffic is left out: it is
noise at this level.
"""

from langchain_core.messages import BaseMessage

MAX_TURNS = 6
MAX_CHARS = 4_000
PER_MESSAGE_CHARS = 800
LABELS = {"human": "User", "ai": "Assistant"}


def recent_conversation(messages: list[BaseMessage], exclude_last: bool = True) -> str:
    """Last MAX_TURNS user/assistant messages as text. Empty string if there are none."""
    chat = [m for m in messages if m.type in LABELS and isinstance(m.content, str) and m.content]
    if exclude_last and chat:
        chat = chat[:-1]  # the latest request is passed separately
    lines = [f"{LABELS[m.type]}: {m.content[:PER_MESSAGE_CHARS]}" for m in chat[-MAX_TURNS:]]
    text = "\n".join(lines)
    return text[-MAX_CHARS:]


def latest_request(messages: list[BaseMessage]) -> str:
    """The most recent user message."""
    humans = [m for m in messages if m.type == "human"]
    return str(humans[-1].content) if humans else ""
