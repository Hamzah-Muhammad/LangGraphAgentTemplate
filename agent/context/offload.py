# agent/context/offload.py  |  BLOCK 4 CONTEXT WINDOW
"""
LARGE TOOL RESULT OFFLOAD (part of BLOCK 4: CONTEXT WINDOW)

The most-cited context rule of 2026: never inline a huge tool result. When a tool
returns more than OFFLOAD_CHARS characters, the full text is written to OFFLOAD_DIR and
the model sees only a short notice plus a preview. It can page through the rest with the
read_file tool (agent/tools/files.py) if and when it needs to.
"""

import re
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

PREVIEW_LINES = 10
PREVIEW_CHARS = 1_000


class OffloadLargeToolResults(AgentMiddleware):
    def __init__(self, max_chars: int, directory: str) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.directory = Path(directory)

    def _offload(self, result):
        if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
            return result
        content = result.content
        if len(content) <= self.max_chars:
            return result

        self.directory.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{result.name or 'tool'}-{result.tool_call_id}")
        path = self.directory / f"{stem[:120]}.txt"
        path.write_text(content, encoding="utf-8")

        preview = "\n".join(content.splitlines()[:PREVIEW_LINES])[:PREVIEW_CHARS]
        notice = (
            f"[{len(content)} chars saved to {path.name}. "
            f'Read more with read_file(name="{path.name}", offset=0, limit=200).]\n\n'
            f"Preview:\n{preview}"
        )
        return result.model_copy(update={"content": notice})

    def wrap_tool_call(self, request, handler):
        return self._offload(handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._offload(await handler(request))
