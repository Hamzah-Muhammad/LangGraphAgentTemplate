"""
BLOCK 2: TOOLS

One file per tool under tools/. Register each tool here so the graph gets one list.
MCP tools (mcp_servers.json) and subagent tools (agents/) are added at build time.

A tool is a capability, not an instruction. If a tool must never run without a check,
put the check INSIDE the tool as code (or gate it in agent/approval.py). Do not rely on
prompt wording to enforce it. Keep read tools and write tools separate.
"""

from tools.example_tool import example_tool
from tools.files import read_file
from tools.memory_tools import recall, remember
from tools.skills import load_skill

# Add new tools to this list. Order does not matter to the model; descriptions do.
ALL_TOOLS = [
    example_tool,
    remember,
    recall,
    read_file,
    load_skill,
]
