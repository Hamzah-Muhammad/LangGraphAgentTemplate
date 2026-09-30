# LangGraphAgentTemplate

A plug-and-play Python agent template on **LangChain 1.x + LangGraph 1.x**, laid out
around the six building blocks of an agent.

Simple things stay simple: edit a Markdown prompt, drop a tool in a folder, flip an MCP
server on in JSON, add a skill as a Markdown file. Complex work is already wired: five
workflow modes (tool loop, router, planner with parallel workers, evaluator-optimizer,
supervisor with specialists), human approval, per-user long-term memory, context
compaction, large-result offload, retries and fallback, trajectory evals, time travel,
tracing and CI.

## Structure

One folder per building block inside `agent/`. Everything outside `agent/` is how you
run, test or configure it. `tests/test_layout.py` fails if a file under `agent/` is
missing from this tree, so the tree always matches the code.

```
LangGraphAgentTemplate/
├── agent/                          the agent: one folder per building block
│   ├── __init__.py                 map of the blocks
│   │
│   ├── model/                      BLOCK 1  MODEL: the only part that thinks
│   │   ├── __init__.py
│   │   ├── model.py                primary, fallback and grader models from .env
│   │   └── structured.py           structured(): portable tool-calling output + 1 retry
│   │
│   ├── tools/                      BLOCK 2  TOOLS: what the agent can do
│   │   ├── __init__.py             ALL_TOOLS: register every Python tool here
│   │   ├── example_tool.py         placeholder tool to copy
│   │   ├── memory_tools.py         remember / recall, per user (writes to BLOCK 5)
│   │   ├── files.py                read_file, only this thread's offloaded results (BLOCK 4)
│   │   ├── skills.py               load_skill, pulls a skill body (BLOCK 3)
│   │   └── mcp.py                  loads servers from mcp_servers.json as tools
│   │
│   ├── prompt/                     BLOCK 3  SYSTEM PROMPT: what the agent is told
│   │   ├── __init__.py
│   │   ├── system.md               rules for every turn; edit this first
│   │   ├── loader.py               system.md + skills index = the system prompt
│   │   ├── skills.py               finds SKILL.md files, builds the one-line index
│   │   └── skills/
│   │       └── write-report/
│   │           └── SKILL.md        placeholder skill to copy
│   │
│   ├── context/                    BLOCK 4  CONTEXT WINDOW: what the model sees
│   │   ├── __init__.py
│   │   ├── compaction.py           clear old tool results at 50%, summarize at 60%, todos
│   │   ├── offload.py              tool results over OFFLOAD_CHARS go to a per-thread file
│   │   └── history.py              recent turns as text, for planner, router, synthesizer
│   │
│   ├── memory/                     BLOCK 5  MEMORY: what the agent remembers
│   │   ├── __init__.py
│   │   └── store.py                SQLite checkpoints per thread + store per user
│   │
│   ├── orchestration/              BLOCK 6  ORCHESTRATION: what runs next
│   │   ├── __init__.py
│   │   ├── graph.py                build_simple_agent (all middleware) + mode dispatch
│   │   ├── reliability.py          call cap, retries, fallback, tool-error recovery
│   │   ├── approval.py             human gate: tools listed here pause first
│   │   ├── studio.py               one graph per mode for langgraph.json
│   │   ├── patterns/
│   │   │   ├── __init__.py
│   │   │   ├── planner.py          plan -> workers -> review (re-plan gaps) -> synthesize
│   │   │   ├── router.py           rules -> model -> one path
│   │   │   ├── evaluator.py        draft -> separate grader -> revise -> human
│   │   │   └── supervisor.py       coordinator -> specialist -> coordinator
│   │   └── subagents/
│   │       ├── __init__.py         SUBAGENT_BUILDERS: subagents offered as tools
│   │       ├── specialist.py       one subagent wrapped as a tool
│   │       └── team.py             researcher + writer for supervisor mode
│   │
│   └── shared/                     not a block: helpers every block uses
│       ├── __init__.py
│       ├── settings.py             every cap, threshold and flag, read from .env
│       ├── runtime.py              per-run context (user_id)
│       ├── schemas.py              structured outputs: Plan, Verdict, Answer
│       └── usage.py                tokens and cost per turn
│
├── run.py                          CLI: --mode --thread --user --once --history --replay
├── mcp_servers.json                MCP servers to load (enabled: true / false)
├── langgraph.json                  LangGraph Studio / Platform entry points
├── pyproject.toml                  dependencies, ruff, pytest
├── .env.example                    every setting, with defaults
├── evals/                          cases.jsonl (answer + tool path), runner, `python -m evals`
├── tests/                          fake-model tests, no key or network needed
├── docs/PATTERNS.md                when to use each mode, its cost, how it fails
└── .github/workflows/ci.yml        ruff + pytest on every push
```

**How the blocks connect.** `agent/orchestration/graph.py` is the only place they meet.
`build_simple_agent` takes the model (1), the tools (2) and the prompt (3), wraps them in
the context (4) and reliability middleware, and attaches memory (5). Every mode in
`patterns/` is built from `build_simple_agent`, so every mode gets all six blocks.

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
| `MAX_WORKERS`, `MAX_CONCURRENCY`, `MAX_PLAN_ROUNDS` | Planner: workers per round, workers calling the model at once, plan/review rounds. |
| `MAX_EVAL_ROUNDS`, `MAX_SUPERVISOR_HOPS` | Caps for the evaluator and supervisor. |
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

**Approval.** Tools named in `agent/orchestration/approval.py` pause before running. Answer `y`, `n`, or
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
| behaviour, tone, standing rules | `agent/prompt/system.md` | no |
| a reusable procedure | add `agent/prompt/skills/<name>/SKILL.md` | no |
| add an MCP server's tools | `mcp_servers.json`, set `enabled: true` | no |
| add a regression case | a line in `evals/cases.jsonl` | no |
| caps, thresholds, feature flags | `.env` | no |
| add a Python tool | copy `agent/tools/example_tool.py`, register in `agent/tools/__init__.py` | small |
| require a human yes for a tool | add its name to `INTERRUPT_ON` in `agent/orchestration/approval.py` | one line |
| add a specialist | edit `agent/orchestration/subagents/team.py` (supervisor) or copy `agent/orchestration/subagents/specialist.py` (tool) | small |
| change routing rules | `DEFAULT_RULES` in `agent/orchestration/patterns/router.py` | one line |
| typed final answers | uncomment `response_format=` in `agent/orchestration/graph.py` | one line |
| a new control flow | copy the closest file in `agent/orchestration/patterns/` | yes |

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
