# agent/tools/example_tool.py  |  BLOCK 2 TOOLS
"""
Placeholder tool. Copy this file to make a real one.

The model sees three things:
  1. the function name
  2. the docstring (this is the tool description; be precise about WHEN to use it)
  3. the argument names, types and defaults

It never sees the body. The body is where validation and enforcement live.
"""

from langchain.tools import tool


@tool
def example_tool(query: str) -> str:
    """Echo a query back. Use this when the user asks you to test the tool loop."""
    # --- validation belongs here, in code ---
    if not query.strip():
        return "error: empty query"

    # --- real work would go here (API call, file read, DB query) ---
    return f"example_tool received: {query!r}"
