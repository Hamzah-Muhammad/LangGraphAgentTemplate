"""
BLOCK 2: TOOLS (Python tools here, MCP servers via mcp.py)

One file per tool in this folder. Register each tool here so the graph gets one list.
MCP tools (mcp_servers.json) and subagent tools (agent/orchestration/subagents/) are
added at build time.

A tool is a capability, not an instruction. If a tool must never run without a check,
put the check INSIDE the tool as code, or gate it in agent/orchestration/approval.py.
Do not rely on prompt wording to enforce it. Keep read tools and write tools separate.
"""

from agent.tools.example_tool import example_tool
from agent.tools.files import read_file
from agent.tools.memory_tools import recall, remember
from agent.tools.skills import load_skill

# Add new tools to this list. Order does not matter to the model; descriptions do.
ALL_TOOLS = [
    example_tool,
    remember,
    recall,
    read_file,
    load_skill,
]
