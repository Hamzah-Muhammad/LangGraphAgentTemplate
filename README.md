# LangGraphAgentTemplate

> **This is a template, not an agent.** It does nothing useful out of the box. It is an
> optimized starting point that a developer, or a coding agent such as Claude Code, copies
> and fills in to build a real agent on **LangChain 1.x + LangGraph 1.x**.

**What you get.** The hard, easy-to-get-wrong parts of an agent, already built and tested,
laid out around the six building blocks of an agent: five workflow modes (tool loop,
router, planner with parallel workers, evaluator-optimizer, supervisor with specialists),
human approval, per-user long-term memory, context compaction, large-result offload,
retries, fallback and timeouts, trajectory evals, time travel, a secret guard, tracing
and CI. It runs on any OpenAI-compatible model or on Claude through a Claude login.

**What you add.** Everything that makes it your agent: its purpose and rules
(`agent/prompt/system.md`), its tools (`agent/tools/`), its procedures
(`agent/prompt/skills/`), and the eval cases that define "working" (`evals/cases.jsonl`).

**What is a placeholder.** The system prompt, `example_tool`, the `write-report` skill,
the researcher and writer team, and the eval cases are stand-ins that show the shape.
Replace them; do not ship them.

## Build an agent from this template

The order a developer or a coding agent should follow. Each step names the one place to
change, so nothing else needs touching.

1. **Copy the template** into a new repo, run `python scripts/setup.py` once (turns on the
   secret guard, creates `.env`), then the tests: `python -m pytest -q`. They need no key
   and must pass before you change anything.
2. **Pick a model** in `.env` (see "Choose your model").
3. **Write the purpose** in `agent/prompt/system.md`: what the agent is for, its rules,
   its tone. Keep enforcement out of the prompt; that belongs in tools and approval.
4. **Add tools**: one file per tool in `agent/tools/`, registered in
   `agent/tools/__init__.py`. Delete `example_tool`. Put validation inside the tool.
5. **Gate risky tools**: list any tool that writes, spends or sends in
   `agent/orchestration/approval.py`.
6. **Pick a workflow mode** using `docs/PATTERNS.md`. Start with `simple`; move up only
   when an eval shows it losing.
7. **Replace the eval cases** in `evals/cases.jsonl` with real requests your agent must
   handle, including the tool path it should take. Run `python -m evals`.
8. **Remove what you do not use**: unused modes, the placeholder skill and team.

**Rules for a coding agent.** They live in [`AGENTS.md`](AGENTS.md), the file coding
agents load automatically when they open a repo. It holds eight hard rules, each pinned by
a test, plus the commands to run after every change. A developer should read it too.
Claude Code reads it natively from version 2.1.277; on an older version, tell it to read
`AGENTS.md` first.

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
│   │   ├── structured.py           structured(): portable tool-calling output + 1 retry
│   │   └── claude_code.py          Claude via your Claude Pro/Max login, no API key
│   │
│   ├── tools/                      BLOCK 2  TOOLS: what the agent can do
│   │   ├── __init__.py             ALL_TOOLS: register every Python tool here
│   │   ├── example_tool.py         placeholder tool to copy
│   │   ├── memory_tools.py         remember / recall / forget, per user (writes to BLOCK 5)
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
│   │   ├── store.py                SQLite checkpoints per thread + store per user
│   │   └── inject.py               adds the user's saved facts to the system prompt
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
├── AGENTS.md                       rules and commands for coding agents (loaded automatically)
├── scripts/check_secrets.py        secret guard: blocks keys and .env files from git
├── .githooks/pre-commit            runs the secret guard before every commit
├── evals/                          cases.jsonl (answer + tool path), runner, `python -m evals`
├── tests/                          fake-model tests, no key or network needed
├── docs/PATTERNS.md                when to use each mode, its cost, how it fails
├── docs/DEPLOY.md                  server, auth, Postgres, build, pre-launch checklist
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
copy .env.example .env
```

Then choose your model (next section). Everything else in `.env` has a working default.

**Keys stay out of git.** Your keys go in `.env`, which git ignores. `.env.example` ships
with every key blank. Turn on the commit guard once per clone, so a key or a `.env` file
is refused before a commit is even created:

```powershell
python scripts/setup.py          # same as: git config core.hooksPath .githooks
```

The same check runs in CI (`tests/test_no_secrets.py`), and you can run it yourself:
`python scripts/check_secrets.py` (add `--history` to scan every commit).

## Choose your model

One line in `.env` decides where the model comes from: **`MODEL_PROVIDER`**.

| | Option A: API-style LLM | Option B: Claude login |
|---|---|---|
| `MODEL_PROVIDER` | `openai` (the default) | `claude-code` |
| What you need | An API key on any OpenAI-compatible host: Groq, NVIDIA NIM, OpenRouter, Together, OpenAI, or a local vLLM / Ollama server | A Claude Pro or Max plan, with Claude Code installed and signed in |
| Lines to fill in | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME` | `CLAUDE_MODEL` (`haiku`, `sonnet`, `opus` or a full model id) |
| Extra install | none | `.venv\Scripts\pip install -e ".[claude]"` |
| Who bills you | the host, per token | nobody extra: it draws from your Claude plan limits |
| Replies | streamed token by token | arrive whole |

### Switch from an API-style LLM to Claude

1. Install the extra once: `.venv\Scripts\pip install -e ".[claude]"`
2. Sign in once: run `claude`, then `/login`.
3. In `.env`, change one line and pick a model:

   ```
   MODEL_PROVIDER=claude-code
   CLAUDE_MODEL=sonnet
   ```

The `OPENAI_*` lines can stay; they are ignored while the provider is `claude-code`.

### Switch back to an API-style LLM

Change the one line back. The `OPENAI_*` lines must be filled in.

```
MODEL_PROVIDER=openai
```

### Switch for one run only

No file edit needed. The flags override `.env` for that run:

```powershell
.venv\Scripts\python run.py --provider claude-code --model haiku
.venv\Scripts\python run.py --provider openai --model llama-3.3-70b-versatile
```

### Check which one is active

Every start prints it on the first line:

```
[model] primary = claude-code: sonnet (Claude login, no API key)
[model] primary = openai-compatible: llama-3.3-70b-versatile @ api.groq.com
```

### Mix them

There are three model roles, and each has its own switch with the same two values:

| Role | Switch | Model lines | Used for |
|---|---|---|---|
| Primary | `MODEL_PROVIDER` | `OPENAI_*` or `CLAUDE_MODEL` | every normal call |
| Fallback | `FALLBACK_PROVIDER` | `FALLBACK_API_KEY`, `FALLBACK_BASE_URL`, `FALLBACK_MODEL_NAME` | one call, when the primary keeps failing |
| Grader | `GRADER_PROVIDER` | `GRADER_API_KEY`, `GRADER_BASE_URL`, `GRADER_MODEL_NAME` | grading drafts in evaluator mode |

For a Claude fallback or grader, set its switch to `claude-code` and put the Claude model
in its `*_MODEL_NAME` line; the key and URL lines are not needed. Example: a free Groq
model as primary with Claude grading.

```
MODEL_PROVIDER=openai
GRADER_PROVIDER=claude-code
GRADER_MODEL_NAME=sonnet
```

The switch is read once at startup. Nothing changes provider in the middle of a run,
except the fallback taking over a failed call.

### How the Claude option works

Calls go through the official Claude Agent SDK and the Claude Code CLI
(`agent/model/claude_code.py`). The template stays in charge: Claude Code's own tools, your
MCP servers and your machine settings are switched off, Claude only decides which template
tool to call, and the template runs it. So approval, timeouts, retries and evals work as
with any other model.

Anthropic's rules for subscription use with the Agent SDK have changed before. If this
stops working, check https://support.claude.com/en/articles/15036540 and switch
`MODEL_PROVIDER` back to `openai`.

## All settings

| `.env` key | Purpose |
|---|---|
| `MODEL_PROVIDER`, `FALLBACK_PROVIDER`, `GRADER_PROVIDER` | `openai` or `claude-code`. See "Choose your model". |
| `CLAUDE_MODEL` | With `claude-code`: `haiku`, `sonnet`, `opus` or a full model id. |
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
| `REQUEST_TIMEOUT_S`, `TOOL_TIMEOUT_S` | Deadline for one model request and for one tool call. |
| `MEMORY_INJECT`, `MEMORY_INJECT_LIMIT` | Put the user's newest saved facts into the system prompt, and how many. |
| `MEMORY_FACT_MAX_CHARS`, `MEMORY_REQUIRE_APPROVAL` | Longest fact `remember` accepts; `true` makes `remember` and `forget` wait for a human. |
| `MAX_MODEL_CALLS` | Hard cap on model calls in one run of one agent (default 25). |
| `MCP_STRICT` | `true` makes a failing MCP server stop startup instead of being skipped. |
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

This is a template for building agents, not a finished agent. Template code, no warranty. Any tool or MCP server you enable runs with your local
permissions. Review tools before wiring them into an agent that can act for you.
