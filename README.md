# LangGraphAgentTemplate

A plug-and-play Python agent shell on **LangChain 1.x + LangGraph 1.x**, laid out around
the six building blocks of an agent. Simple things stay simple: edit a Markdown prompt,
drop a tool in a folder, flip an MCP server on in JSON. Complex things are already wired:
a planner that fans work out to parallel workers, subagents as tools, human approval,
long-term memory per user, retries and fallback, evals, tracing, CI.

## The six blocks

| Block | File | Library |
|---|---|---|
| Model | `agent/model.py` | LangChain `init_chat_model`, any OpenAI-compatible host, optional fallback model |
| Tools | `tools/` + `mcp_servers.json` + `agents/` | LangChain `@tool`, MCP servers via `langchain-mcp-adapters`, subagents wrapped as tools |
| System Prompt | `prompts/system.md` | plain Markdown, read at startup |
| Context Window | `agent/context.py` | `before_model` hook + `SummarizationMiddleware` |
| Memory | `agent/memory.py`, `tools/memory_tools.py` | LangGraph SQLite checkpointer (per thread) + Store (per user, cross-thread) |
| Orchestration | `agent/graph.py` | `create_agent` (simple) and a hand-written `StateGraph` planner (complex) |

Cross-cutting: `agent/reliability.py` (call limit, retries, fallback, tool-error recovery),
`agent/approval.py` (human-in-the-loop gate), `agent/runtime.py` (per-run context such as
`user_id`), `agent/schemas.py` (structured output), `agent/usage.py` (tokens and cost).

## Setup

```powershell
cd C:\Apps\LangGraphAgentTemplate
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env      # then fill in the values
```

| `.env` key | Purpose |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME` | primary model on any OpenAI-compatible host (Groq, NVIDIA NIM, OpenRouter, local vLLM/Ollama) |
| `FALLBACK_*` (same three) | optional second model used after retries fail |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | optional; set the key and tracing turns on |
| `MEMORY_DB_PATH` | SQLite file for checkpoints and long-term memory |
| `DEFAULT_USER_ID` | namespace for long-term memory when `--user` is not given |
| `PRICE_IN_PER_M`, `PRICE_OUT_PER_M` | optional USD per 1M tokens for the cost line |

## Run

```powershell
.venv\Scripts\python run.py                            # simple agent, streams tokens
.venv\Scripts\python run.py --mode planner             # plan -> parallel workers -> synthesize
.venv\Scripts\python run.py --thread topic --user bob  # own memory thread, own user namespace
.venv\Scripts\python run.py --once "hello"             # single turn
```

Each turn ends with a usage line (tokens, model calls, cost if prices are set).
Restart with the same `--thread` and the conversation resumes from `memory.db`.

**Approval:** tools named in `agent/approval.py` pause the run before executing. You see the
call and answer `y` (approve), `n` (reject) or `e` (edit the args as JSON). The graph resumes
from the saved checkpoint.

**Visual debugger:** `langgraph dev` opens LangGraph Studio on the same graph
(`langgraph.json`). The same file deploys it to LangGraph Platform or Docker unchanged.

## Test and evals

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\python -m pytest -q      # fake model, no key, no network
.venv\Scripts\python -m evals          # evals/cases.jsonl against the real model
```

Tests cover: prompt loads, one turn, thread memory, tool call paused at the approval gate
then run, long-term memory per user, planner fan-out, SQLite persistence across a reopen,
and every eval case through the fake model. CI (`.github/workflows/ci.yml`) runs ruff and
pytest on each push.

## Making it yours

| Want to change | Edit | Code change? |
|---|---|---|
| behaviour, tone, rules of thumb | `prompts/system.md` | no |
| add an MCP server's tools | `mcp_servers.json`, set `enabled: true` | no |
| add a Python tool | copy `tools/example_tool.py`, register in `tools/__init__.py` | small |
| require a human yes for a tool | add its name to `INTERRUPT_ON` in `agent/approval.py` | one line |
| add a specialist subagent | copy `agents/specialist.py`, register in `agents/__init__.py` | small |
| typed final answers | uncomment `response_format=` in `agent/graph.py`, edit `agent/schemas.py` | one line |
| trim what the model sees | fill in `trim_context` in `agent/context.py` | small |
| new control flow | copy `build_planner_graph` in `agent/graph.py` | yes, this is the one place |
| add a regression case | append a line to `evals/cases.jsonl` | no |

## Rules baked in

- Nodes never call each other. State in, state out; edges decide what runs next.
- Tools never reason. Validation and permission checks live inside the tool as code.
- Separate read tools from write tools. Never one tool that can both query and delete.
- Every run has a `thread_id` (resume, replay, approval) and a `user_id` (memory namespace).
- Errors reach the model as text, not as crashes, so it can recover or explain.
- Middleware order is load-bearing: first in the list is the outermost wrapper. Fallback wraps
  retry, so the primary is retried before the fallback runs. `tests/test_reliability.py` pins this.
- Parallel branches can pause together. Resume them with one `{interrupt_id: value}` map.

## Disclaimer

Template code, no warranty. Any tool or MCP server you enable runs with your local
permissions. Review tools before wiring them into an agent that can act on your behalf.
