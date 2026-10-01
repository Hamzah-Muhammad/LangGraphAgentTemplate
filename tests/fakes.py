"""Scripted fake chat model for tests. No key, no network."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class FakeToolModel(GenericFakeChatModel):
    """Replays scripted AI messages; accepts bind_tools and with_structured_output.

    `structured` is what with_structured_output returns: a dict (same answer every call)
    or a list of dicts (one per call, in order).
    """

    structured: Any = None

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        if isinstance(self.structured, list):
            answers = iter(self.structured)
            return RunnableLambda(lambda _input: schema(**next(answers)))
        payload = self.structured or {}
        return RunnableLambda(lambda _input: schema(**payload))


def to_message(turn: AIMessage | str | dict, index: int = 0) -> AIMessage:
    """A script turn: plain text, an AIMessage, or {"content": ..., "tool_calls": [...]}."""
    if isinstance(turn, AIMessage):
        return turn
    if isinstance(turn, str):
        return AIMessage(content=turn)
    calls = [
        {"id": tc.get("id", f"call_{index}_{n}"), "name": tc["name"], "args": tc.get("args", {})}
        for n, tc in enumerate(turn.get("tool_calls", []))
    ]
    return AIMessage(content=turn.get("content", ""), tool_calls=calls)


def fake_model(*script: AIMessage | str | dict, structured: Any = None) -> FakeToolModel:
    messages = [to_message(t, i) for i, t in enumerate(script)]
    return FakeToolModel(messages=iter(messages), structured=structured)
