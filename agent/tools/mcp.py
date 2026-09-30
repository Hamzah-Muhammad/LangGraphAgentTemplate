# agent/tools/mcp.py  |  BLOCK 2 TOOLS
"""
MCP TOOLS (part of BLOCK 2: TOOLS)

Loads every server in mcp_servers.json that has "enabled": true and exposes its tools
to the agent, side by side with the Python tools in agent/tools/. Zero code per server.

Add your own MCP servers (MinniMemoryMCP, GitHub, Postgres, ...) by editing the JSON.
"""

import json
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG_PATH = Path(__file__).resolve().parents[2] / "mcp_servers.json"  # repo root


def _enabled_connections(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        return {}
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    connections = {}
    for name, entry in raw.items():
        if name.startswith("_") or not isinstance(entry, dict) or not entry.get("enabled"):
            continue
        connections[name] = {k: v for k, v in entry.items() if k != "enabled"}
    return connections


async def load_mcp_tools(config_path: Path = CONFIG_PATH) -> list[BaseTool]:
    """Return tools from all enabled MCP servers. Empty list if none are enabled."""
    connections = _enabled_connections(config_path)
    if not connections:
        return []
    client = MultiServerMCPClient(connections)
    return await client.get_tools()
