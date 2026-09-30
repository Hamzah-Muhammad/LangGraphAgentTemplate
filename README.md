# LangGraphAgentTemplate

A plug-and-play Python agent shell built on **LangChain 1.x + LangGraph 1.x**, laid out
around the six building blocks of an agent. Every block lives in its own file, so you
change one thing at a time and never touch the loop unless the control flow itself changes.

| Block | File | Library |
|---|---|---|
| Model | `agent/model.py` | LangChain `init_chat_model` |
| Tools | `tools/` | LangChain `@tool` |
| System Prompt | `prompts/system.md` | plain Markdown, read at startup |
| Context Window | `agent/context.py` | LangChain middleware (`before_model`, summarization) |
| Memory | `agent/memory.py` | LangGraph checkpointer (SQLite) + Store |
| Orchestration | `agent/graph.py` | LangGraph `create_agent` |

Plus: `agent/approval.py` (human-in-the-loop gate), `run.py` (CLI), `tests/` (smoke test with a fake model, no API key needed).

## Setup

```powershell
cd C:\Apps\LangGraphAgentTemplate
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then fill in the values
```

`.env` keys:

| Key | Purpose |
|---|---|
| `OPENAI_API_KEY` | key for any OpenAI-compatible host (Groq, NVIDIA NIM, OpenRouter, a local server) |
| `OPENAI_BASE_URL` | that host's base URL, e.g. `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | model id on that host |
| `LANGSMITH_API_KEY` | optional. Set it and tracing turns on. Leave blank and it stays off. |
| `LANGSMITH_PROJECT` | optional project name in LangSmith |

## Run

```powershell
.venv\Scripts\python run.py                       # interactive chat, thread "default"
.venv\Scripts\python run.py --thread my-topic     # separate memory thread
.venv\Scripts\python run.py --once "hello"        # single turn
```

Memory is per `--thread`. Restart with the same thread and the agent picks up where it left off (SQLite file `memory.db`, ignored by git).

When the agent calls a tool listed in `agent/approval.py`, the run pauses and asks you to approve, edit or reject the call before it executes.

## Test

```powershell
.venv\Scripts\python -m pytest -q
```

The test builds the whole graph with a fake chat model, so it proves the wiring without a key or network.

## Making it yours

1. **Behaviour**: edit `prompts/system.md`. No code change.
2. **Capability**: add a `@tool` function in `tools/`, register it in `tools/__init__.py`.
3. **Guardrail**: add the tool's name to `INTERRUPT_ON` in `agent/approval.py` if it needs a human yes.
4. **Control flow**: only if `create_agent` cannot express it (planner/worker split, custom routing), replace the body of `agent/graph.py` with a hand-written `StateGraph`. Everything else stays.

## Rules baked in

- Nodes never call each other. State in, state out; edges decide what runs next.
- Tools never reason. Validation and permission checks live inside the tool as code.
- Every run has a `thread_id`. That is what gives you resume, replay and approval for free.

## Disclaimer

Template code, no warranty. Any tool you add runs with your local permissions. Review tools before wiring them into an agent that can act on your behalf.
