# agent/model/claude_code.py  |  BLOCK 1 MODEL
"""
CLAUDE THROUGH YOUR CLAUDE SUBSCRIPTION (no API key)

ChatClaudeCode makes the Claude Agent SDK look like a normal LangChain chat model. The
SDK runs the Claude Code CLI, which uses the login you already have (`claude` then
`/login`), so calls draw from your Pro/Max plan limits instead of an API bill.

Set MODEL_PROVIDER=claude-code in .env. Needs: `pip install -e ".[claude]"` and the
Claude Code CLI installed and logged in.

DESIGN: the template stays in charge of tools.
  Claude Code has its own agent loop and its own tools (file edits, shell). All of that
  is switched OFF here. Claude only decides which template tool to call; the template's
  own loop runs it. So the approval gate, timeouts, retries, offload and evals work
  exactly as with any other model.

  How a tool call travels:
    1. Each template tool is registered with Claude Code as a STUB with the real name,
       description and argument schema, so Claude uses its native tool calling.
    2. A PreToolUse hook answers "defer" to every tool request. That ends the run on the
       spot and hands the request back to us. The stub never executes.
    3. The request becomes an AIMessage with `tool_calls`, which LangGraph executes
       through the template's middleware. Several tools requested at once all come through.

  An earlier design asked Claude to write tool calls as JSON in a structured answer.
  A real run showed Claude often tried to call the tool natively instead, so it was
  replaced by the mechanism above (verified 2026-09-30, SDK 0.2.163, Haiku 4.5).

  Plans, routes and verdicts use the SDK's native structured output (`output_format`),
  which costs one extra SDK turn.

ISOLATION: each call starts a clean Claude Code process with no built-in tools, none of
the user's MCP servers, no user/project settings (so no CLAUDE.md, hooks or memory files
leak in) and a temp working directory.

LIMITS: no token streaming (the reply arrives whole), one CLI process per call (about a
second of overhead), and usage shares your subscription limits. Anthropic's policy on
subscription use with the Agent SDK has changed before; check it if this stops working:
https://support.claude.com/en/articles/15036540
"""

import asyncio
import json
import tempfile
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

TEXT_TURNS = 1  # a text reply, or a deferred tool request, both end on turn 1
STRUCTURED_TURNS = 4  # thinking + the StructuredOutput call, with slack for one slip
STUB_SERVER = "host"
STUB_PREFIX = f"mcp__{STUB_SERVER}__"
LABELS = {"human": "User", "ai": "Assistant"}


class ClaudeCodeError(RuntimeError):
    """The Claude Code CLI returned an error (auth, rate limit, max turns...)."""


def render_conversation(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    """(system text, prompt text). The SDK takes one prompt string, so the history is
    rendered as a transcript. A lone user message is passed through untouched."""
    system = "\n\n".join(str(m.content) for m in messages if m.type == "system")
    turns = [m for m in messages if m.type != "system"]
    if len(turns) == 1 and turns[0].type == "human":
        return system, str(turns[0].content)

    lines = []
    for m in turns:
        if m.type == "tool":
            lines.append(f"Tool result ({m.name or 'tool'}, id {m.tool_call_id}):\n{m.content}")
            continue
        text = str(m.content) if m.content else ""
        for call in getattr(m, "tool_calls", None) or []:
            args = json.dumps(call["args"])
            text += f"\n[requested tool {call['name']} id {call['id']} args {args}]"
        lines.append(f"{LABELS.get(m.type, m.type)}: {text.strip()}")
    transcript = "\n\n".join(lines)
    return system, (
        f"<conversation>\n{transcript}\n</conversation>\n\n"
        "Write the assistant's next turn in this conversation. Tool results shown above "
        "have already happened; do not request the same call again."
    )


async def _defer(input_data, tool_use_id, context):
    """PreToolUse hook: hand every tool request back to the host instead of running it."""
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "defer"}}


def _stub_server(tools: list[dict]):
    """Claude-facing copies of the template's tools. Real schema, no behaviour."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    def stub(spec: dict):
        f = spec["function"]
        schema = f.get("parameters") or {"type": "object", "properties": {}}

        @tool(f["name"], f.get("description", ""), schema)
        async def never_runs(args):  # the defer hook stops the run before this is reached
            return {"content": [{"type": "text", "text": "deferred to host"}]}

        return never_runs

    return create_sdk_mcp_server(STUB_SERVER, tools=[stub(t) for t in tools])


def build_options(model: str, system: str, schema: dict | None, tools: list[dict] | None = None):
    """Isolated Claude Code options. Imported lazily: the SDK is an optional extra."""
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    tools = tools or []
    return ClaudeAgentOptions(
        model=model,
        system_prompt=system or "You are a helpful assistant.",
        tools=[],  # no built-in tools: the template runs tools, not Claude Code
        mcp_servers={STUB_SERVER: _stub_server(tools)} if tools else {},
        strict_mcp_config=True,  # ignore the user's own MCP servers
        allowed_tools=[STUB_PREFIX + t["function"]["name"] for t in tools],
        hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_defer])]} if tools else None,
        setting_sources=[],  # no CLAUDE.md, hooks or settings from the machine
        cwd=tempfile.gettempdir(),
        max_turns=STRUCTURED_TURNS if schema else TEXT_TURNS,
        output_format={"type": "json_schema", "schema": schema} if schema else None,
    )


async def sdk_call(
    prompt: str, system: str, model: str, schema: dict | None, tools: list[dict] | None = None
) -> dict:
    """One Claude Code run. Returns {"text", "structured", "usage", "tool_calls"}.
    The single seam between this file and the SDK: tests replace it and never start
    the CLI."""
    from claude_agent_sdk import query

    text, structured, usage, error, calls = "", None, {}, None, []
    try:
        options = build_options(model, system, schema, tools)
        async for message in query(prompt=prompt, options=options):
            kind = type(message).__name__
            if kind == "AssistantMessage":
                for block in message.content:
                    if type(block).__name__ == "ToolUseBlock":
                        calls.append({
                            "name": block.name.removeprefix(STUB_PREFIX),
                            "args": block.input or {},
                            "id": block.id,
                        })
            elif kind == "ResultMessage":
                text, structured = message.result or "", message.structured_output
                usage = message.usage or {}
                if message.is_error:
                    error = "; ".join(message.errors or []) or message.subtype
    except Exception as exc:  # the SDK raises after an error result; keep its message
        error = error or f"{type(exc).__name__}: {exc}"
    if error:
        raise ClaudeCodeError(error)
    return {"text": text, "structured": structured, "usage": usage, "tool_calls": calls}


def usage_metadata(usage: dict) -> dict:
    tokens_in = sum(usage.get(k, 0) or 0 for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    tokens_out = usage.get("output_tokens", 0) or 0
    return {"input_tokens": tokens_in, "output_tokens": tokens_out,
            "total_tokens": tokens_in + tokens_out}


class ChatClaudeCode(BaseChatModel):
    """LangChain chat model backed by the Claude Agent SDK and your Claude login."""

    model: str = "sonnet"
    timeout_s: float = 120.0
    bound_tools: list[dict] = Field(default_factory=list)
    structured_schema: dict | None = None  # set by with_structured_output

    @property
    def _llm_type(self) -> str:
        return "claude-code"

    # -- tools -----------------------------------------------------------------
    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound_tools": [convert_to_openai_tool(t) for t in tools]})

    # -- structured output (plans, routes, verdicts) -----------------------------
    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        """Native SDK structured output. `method=` is accepted and ignored.

        The call still goes through the normal chat-model path (a copy of this model with
        `structured_schema` set), so callbacks fire and the usage counter sees it. A direct
        SDK call here once made the planner report 2 model calls instead of 4."""
        json_schema = schema if isinstance(schema, dict) else schema.model_json_schema()
        bound = self.model_copy(update={"structured_schema": json_schema, "bound_tools": []})

        def parse(message: AIMessage):
            data = json.loads(str(message.content))
            return data if isinstance(schema, dict) else schema.model_validate(data)

        return bound | RunnableLambda(parse)

    # -- generation --------------------------------------------------------------
    async def _call(self, prompt, system, schema, tools) -> dict:
        try:
            return await asyncio.wait_for(
                sdk_call(prompt, system, self.model, schema, tools), self.timeout_s)
        except TimeoutError:
            raise ClaudeCodeError(f"no reply within {self.timeout_s:g}s") from None

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        system, prompt = render_conversation(messages)
        if self.structured_schema:
            result = await self._call(prompt, system, self.structured_schema, None)
            if result["structured"] is None:
                raise ClaudeCodeError("no structured output returned")
            result = {**result, "text": json.dumps(result["structured"]), "tool_calls": []}
        else:
            result = await self._call(prompt, system, None, self.bound_tools or None)
        calls = [{**c, "type": "tool_call"} for c in result.get("tool_calls") or []]
        message = AIMessage(
            content="" if calls else result["text"],
            tool_calls=calls,
            usage_metadata=usage_metadata(result["usage"]),
            response_metadata={"model_name": f"claude-code:{self.model}"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Sync callers (rare; the template is async). Must not be inside a running loop.
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))


def _as_messages(value: Any) -> list[BaseMessage]:
    """with_structured_output callers pass a list of messages or a prompt value."""
    if hasattr(value, "to_messages"):
        return value.to_messages()
    if isinstance(value, str):
        return [HumanMessage(value)]
    return list(value)
