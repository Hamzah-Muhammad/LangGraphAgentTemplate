"""Scripted fake chat model for tests. No key, no network."""

from collections.abc import Iterable

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class FakeToolModel(GenericFakeChatModel):
    """Replays scripted AI messages; accepts bind_tools and with_structured_output."""

    structured: dict | None = None  # kwargs for the schema returned by with_structured_output

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        payload = self.structured or {}
        return RunnableLambda(lambda _input: schema(**payload))


def fake_model(*scripted: AIMessage | str, structured: dict | None = None) -> FakeToolModel:
    msgs: Iterable[AIMessage] = (
        m if isinstance(m, AIMessage) else AIMessage(content=m) for m in scripted
    )
    return FakeToolModel(messages=iter(list(msgs)), structured=structured)
