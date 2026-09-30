# LangGraphAgentTemplate

A plug-and-play Python agent template on **LangChain 1.x + LangGraph 1.x**, laid out
around the six building blocks of an agent.

Simple things stay simple: edit a Markdown prompt, drop a tool in a folder, flip an MCP
server on in JSON, add a skill as a Markdown file. Complex work is already wired: five
workflow modes (tool loop, router, planner with parallel workers, evaluator-optimizer,
supervisor with specialists), human approval, per-user long-term memory, context
compaction, large-result offload, retries and fallback, trajectory evals, time travel,
tracing and CI.

## The six blocks

| Block | File | What it does |
|---|---|---|
| Model | `agent/model.py` | Any OpenAI-compatible host from `.env`. Optional fallback and grader models. |
| Tools | `tools/`, `mcp_servers.json`, `agents/` | Python `@tool` functions, MCP servers with no code, subagents wrapped as tools. |
| System Prompt | `prompts/system.md`, `skills/` | Markdown read at startup, plus a one-line index of skills loaded on demand. |
| Context Window | `agent/context.py`, `agent/offload.py` | Clears old tool results at 50% of the window, summarizes at 60%, offloads huge tool results to files. |
| Memory | `agent/memory.py`, `tools/memory_tools.py` | SQLite checkpoints per thread, and a per-user store the agent reads and writes. |
| Orchestration | `agent/graph.py`, `agent/patterns/` | Five modes built from the same blocks. |

Cross-cutting: `agent/reliability.py` (call cap, retries, fallback, tool-error recovery),
`agent/approval.py` (human gate), `agent/settings.py` (every tunable number),
`agent/runtime.py` (per-run `user_id`), `agent/schemas.py` (structured decisions),
`agent/usage.py` (tokens and cost).

## Workflow modes

| Mode | Use when |
|---|---|
| `simple` | Default. One goal, tools as needed. |
| `router` | Requests come in different kinds. Rules first, model second. Ships routing between `simple` and `planner`. |
| `planner` | Breadth-first work with several independent parts. Parallel workers. |
| `evaluator` | Output has checkable quality criteria. Separate grader, max 3 rounds, then a human. |
| `supervisor` | Distinct roles. A coordinator delegates to a researcher and a writer. |

`docs/PATTERNS.md` covers when to use each, what each costs in model calls, how each
fails, and the rules each one enforces in code.

## Setup

```powershell
cd C:\Apps\LangGraphAgentTemplate
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env      # then fill in the values
```

| `.env` key | Purpose |
|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME` | Primary model on any OpenAI-compatible host (Groq, NVIDIA NIM, OpenRouter, local vLLM or Ollama). |
| `FALLBACK_*` (same three) | Optional second model used after retries fail. |
| `GRADER_*` (same three) | Optional separate model for the evaluator's grader. |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | Optional. Set the key and tracing turns on. |
| `MEMORY_DB_PATH`, `DEFAULT_USER_ID` | SQLite file, and the memory namespace when `--user` is not given. |
| `CONTEXT_WINDOW_TOKENS` | Your model's real limit. Compaction thresholds derive from it. |
| `AGENT_TODOS` | `true` adds the `write_todos` planning tool. |
| `OFFLOAD_CHARS`, `OFFLOAD_DIR` | Tool results longer than this are saved to files. |
| `MAX_WORKERS`, `MAX_EVAL_ROUNDS`, `MAX_SUPERVISOR_HOPS` | Caps for the planner, evaluator and supervisor. |
| `PRICE_IN_PER_M`, `PRICE_OUT_PER_M` | Optional USD per 1M tokens for the cost line. |

## Run

```powershell
.venv\Scripts\python run.py                             # simple agent, streams tokens
.venv\Scripts\python run.py --mode router               # rules or model pick the path
.venv\Scripts\python run.py --mode planner --once "compare tea, coffee and mate"
.venv\Scripts\python run.py --mode evaluator            # draft, grade, revise
.venv\Scripts\python run.py --mode supervisor           # coordinator + specialists
.venv\Scripts\python run.py --thread topic --user bob   # own thread, own memory namespace
```

Each turn ends with a usage line. Restart with the same `--thread` and the conversation
resumes from `memory.db`.

**Approval.** Tools named in `agent/approval.py` pause before running. Answer `y`, `n`, or
`e` to edit the arguments as JSON. When the evaluator runs out of rounds it pauses the
same way and you accept or replace the draft.

**Time travel.**

```powershell
.venv\Scripts\python run.py --thread topic --history          # checkpoints, newest first
.venv\Scripts\python run.py --thread topic --replay <id>      # re-run from that point
```

**Visual debugger.** `langgraph dev` opens LangGraph Studio with all five modes as
separate graphs (`langgraph.json`). The same file deploys to LangGraph Platform or Docker.

## Test and evals

```powershell
.venv\Scripts\ruff check .
.venv\Scripts\python -m pytest -q      # fake models, no key, no network
.venv\Scripts\python -m evals          # evals/cases.jsonl against the real model
```

Evals check the outcome and the trajectory: which tools ran, in what order, which must
never run, and how many calls were allowed. CI runs ruff and pytest on every push.

## Making it yours

| Want to change | Edit | Code change? |
|---|---|---|
| behaviour, tone, standing rules | `prompts/system.md` | no |
| a reusable procedure | add `skills/<name>/SKILL.md` | no |
| add an MCP server's tools | `mcp_servers.json`, set `enabled: true` | no |
| add a regression case | a line in `evals/cases.jsonl` | no |
| caps, thresholds, feature flags | `.env` | no |
| add a Python tool | copy `tools/example_tool.py`, register in `tools/__init__.py` | small |
| require a human yes for a tool | add its name to `INTERRUPT_ON` in `agent/approval.py` | one line |
| add a specialist | edit `agents/team.py` (supervisor) or copy `agents/specialist.py` (tool) | small |
| change routing rules | `DEFAULT_RULES` in `agent/patterns/router.py` | one line |
| typed final answers | uncomment `response_format=` in `agent/graph.py` | one line |
| a new control flow | copy the closest file in `agent/patterns/` | yes |

## Rules baked in

- Nodes never call each other. State in, state out; edges decide what runs next.
- Tools never reason. Validation and permission checks live inside the tool as code.
- Separate read tools from write tools. Never one tool that can both query and delete.
- Every loop has a cap in code, and running out escalates to a human or stops cleanly.
- Every decision taken from model output is structured, so it can be logged and tested.
- Every run has a `thread_id` (resume, replay, approval) and a `user_id` (memory namespace).
- Errors reach the model as text, not as crashes, so it can recover or explain.
- Middleware order is load-bearing: first in the list is the outermost wrapper.
  `tests/test_reliability.py` pins the order.

## Disclaimer

Template code, no warranty. Any tool or MCP server you enable runs with your local
permissions. Review tools before wiring them into an agent that can act for you.
