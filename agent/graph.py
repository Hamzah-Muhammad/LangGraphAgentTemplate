"""
BLOCK 6: ORCHESTRATION

The loop. Zero model reasoning lives here; this file only decides what runs next.

create_agent builds the standard graph:

    START -> [middleware.before_model] -> model -> (tool calls?) -> tools -> model ... -> END

When create_agent cannot express your flow (planner/worker, custom routing, parallel
branches), replace build_graph() with a hand-written StateGraph. Every other file stays.
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from agent.approval import build_approval_middleware
from agent.context import build_context_middleware
from tools import ALL_TOOLS

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    """BLOCK 3: SYSTEM PROMPT. Read from Markdown so behaviour is editable without code."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """Assemble the six blocks into one compiled, runnable graph."""
    middleware = [
        *build_context_middleware(model),   # Context Window
        build_approval_middleware(),        # human gate
    ]
    return create_agent(
        model=model,                        # Model
        tools=ALL_TOOLS,                    # Tools
        system_prompt=load_system_prompt(), # System Prompt
        middleware=middleware,
        checkpointer=checkpointer,          # Memory (short-term)
        store=store,                        # Memory (long-term)
        name="template-agent",
    )
