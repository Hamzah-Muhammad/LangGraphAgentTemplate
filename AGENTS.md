# Instructions for coding agents

This repo is a **template** for building complex agents on LangChain 1.x and LangGraph 1.x.
It is not an agent. You are most likely here to build one on top of it. Detail lives in
`README.md` (structure, setup, build steps) and `docs/PATTERNS.md` (which workflow mode).

## Project purpose

PLACEHOLDER. When the agent is built, replace this line with two sentences saying what it
is for, who uses it, and what it must never do.

## Commands

Run after every change. All three must stay green.

    ruff check .
    python -m pytest -q        # fake models, no key, no network
    python -m evals            # real model from .env; run it when behaviour changes

## Hard rules

Each rule is pinned by a test. Break one and a test fails.

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
   a `.env` file. Turn the guard on once per clone: `git config core.hooksPath .githooks`
   (`scripts/check_secrets.py`, `tests/test_no_secrets.py`).
8. **Logs.** Never print a URL, header or raw error that could hold a credential. Print the
   host and the error type only.

## Placeholders to replace, not ship

`agent/prompt/system.md`, `agent/tools/example_tool.py`, `agent/prompt/skills/write-report/`,
`agent/orchestration/subagents/team.py`, and the cases in `evals/cases.jsonl`.

## Done means

Lint clean, tests green, README Structure tree matches the files, and a real run for any
change to prompts, tools or workflow modes. Say what you did not run.
