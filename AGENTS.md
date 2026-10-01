# Instructions for coding agents

This repo is a **template** for building complex agents on LangChain 1.x and LangGraph 1.x.
It is not an agent. You are most likely here to build one on top of it. Detail lives in
`README.md` (structure, setup, build steps) and `docs/PATTERNS.md` (which workflow mode).

## Project purpose

PLACEHOLDER. When the agent is built, replace this line with two sentences saying what it
is for, who uses it, and what it must never do.

## Start here

1. `python scripts/setup.py`, then `python -m pytest -q`. It must pass before you change anything.
2. Ask the user for what the template cannot guess: the agent's purpose, its tools, the model,
   and which actions need human approval. Do not invent them.
3. Follow the eight steps in `README.md` ("Build an agent from this template"), in order.
4. Replace the placeholders below, then run `python -m evals` against the real model.
5. Before launch, work through the checklist in `docs/DEPLOY.md`.

## Commands

Run after every change. All three must stay green.

    ruff check .
    python -m pytest -q        # fake models, no key, no network
    python -m evals            # real model from .env; run it when behaviour changes

## Hard rules

Each rule has a test for the code that exists today. A test cannot see code you add, so
the rule only holds if you follow it. A subagent built the wrong way passes every test.

1. **Layout.** One folder per building block under `agent/`: model, tools, prompt, context,
   memory, orchestration, plus shared. No other folders. A new file starts with the header
   `# agent/<path>  |  BLOCK n NAME` and is added to the README Structure tree
   (`tests/test_layout.py`).
2. **Middleware order.** The order in `build_simple_agent` in
   `agent/orchestration/graph.py` is load-bearing. Read `agent/orchestration/reliability.py`
   before touching it (`tests/test_reliability.py`).
3. **Nested agents.** Build every subagent, worker or specialist with `build_simple_agent`,
   never bare `create_agent`. A bare one skips the approval gate
   (`tests/test_audit_regressions.py`).
4. **Model decisions.** A plan, route, verdict or next-specialist choice taken from model
   output goes through `structured()` in `agent/model/structured.py` and falls back to a
   safe default on a bad reply. It never crashes the run (`tests/test_workflow_audit.py`).
5. **Tools.** Validation lives inside the tool as code. Tools that write, spend or send are
   listed in `agent/orchestration/approval.py`. Keep read tools and write tools separate.
6. **Loops.** Every loop has a cap in code. Per-request state resets at the start of each
   turn (`tests/test_run_hardening.py`).
7. **Secrets.** Keys live only in `.env`, which git ignores. Never commit a key, a token or
   a `.env` file. Turn the guard on once per clone: `python scripts/setup.py`
   (`scripts/check_secrets.py`, `tests/test_no_secrets.py`).
8. **Logs.** Never print a URL, header or raw error that could hold a credential. Print the
   host and the error type only.
9. **Memory.** Saved facts are read into the system prompt, so `remember` is a write path
   into the prompt. Keep it bounded (length cap, one copy per fact) and decide on
   `MEMORY_REQUIRE_APPROVAL`. Memory is per user; never read or write another user's
   namespace. On a server the user comes from auth (`tests/test_hardening.py`).

## Placeholders to replace, not ship

`agent/prompt/system.md`, `agent/tools/example_tool.py`, `agent/prompt/skills/write-report/`,
`agent/orchestration/subagents/team.py`, and the cases in `evals/cases.jsonl`.

## Not verified in this repo

A real run on an OpenAI-compatible host; approval prompts on a real model; the auth handler,
Postgres swap and `langgraph build` recipes in `docs/DEPLOY.md`. Test them before relying on them.

## Done means

Lint clean, tests green, README Structure tree matches the files, and a real run for any
change to prompts, tools or workflow modes. Say what you did not run.
