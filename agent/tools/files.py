# agent/tools/files.py  |  BLOCK 2 TOOLS
"""read_file: page through tool results that agent/context/offload.py saved to disk.

Read-only and jailed to OFFLOAD_DIR/<this thread>. It cannot reach another thread's
results or any other file on the machine.
"""

from pathlib import Path

from langchain.tools import ToolRuntime, tool

from agent.context.offload import thread_folder
from agent.shared.runtime import Context
from agent.shared.settings import get_settings


@tool
def read_file(name: str, runtime: ToolRuntime[Context], offset: int = 0, limit: int = 200) -> str:
    """Read part of a saved tool result. Use when a tool result says it was saved to a file.
    `offset` is the first line (0-based), `limit` the number of lines to return."""
    base = (Path(get_settings().offload_dir) / thread_folder(runtime.config)).resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.is_file():
        return f"error: no saved result named {name!r}"

    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(offset, 0)
    end = min(start + max(limit, 1), len(lines))
    body = "\n".join(lines[start:end])
    return f"{body}\n\n(lines {start}-{end - 1} of {len(lines)})"
