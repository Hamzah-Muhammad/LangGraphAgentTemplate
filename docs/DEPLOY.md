# Going to production

The template runs locally out of the box. This page is the path from there to a server.
Each step says whether it was **verified** in this repo or is a **recipe** nobody ran here.

## 1. Run it as a server (verified)

```powershell
.venv\Scripts\langgraph dev          # local API + Studio, serves every graph in langgraph.json
```

`langgraph.json` lists one graph per mode. The server supplies the checkpointer and store,
so the graphs are built without them (`agent/orchestration/studio.py`).
`tests/test_layout.py` fails if a path in `langgraph.json` stops existing.

## 2. Tell the server who the user is (recipe)

Memory is per user. On a server the user is not `--user`; it comes from the request.
`resolve_user_id` (`agent/memory/store.py`) already reads, in order: `context.user_id`,
the authenticated identity, `configurable.user_id`, then `"anonymous"`. Without one of the
middle two, every caller shares one memory.

Add an auth handler at the repo root (not under `agent/`, whose layout is pinned):

```python
# auth.py
from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def authenticate(authorization: str | None) -> Auth.types.MinimalUserDict:
    user_id = verify_token(authorization)        # YOUR check: JWT, API key lookup, ...
    return {"identity": user_id}
```

then add `"auth": {"path": "./auth.py:auth"}` to `langgraph.json`. The identity becomes
`configurable.langgraph_auth_user`, and the memory namespace follows it. What is tested here
is the lookup order (`tests/test_hardening.py`); the handler itself is yours to write and test.

## 3. Swap SQLite for Postgres (recipe)

SQLite is one file on one machine. For more than one process, use Postgres. Install
`langgraph-checkpoint-postgres` and change `open_memory` in `agent/memory/store.py`:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

async with (
    AsyncPostgresSaver.from_conn_string(url) as checkpointer,
    AsyncPostgresStore.from_conn_string(url) as store,
):
    await checkpointer.setup()
    await store.setup()
    yield checkpointer, store
```

Take `url` from the host's environment. Do not run `langgraph dev` against it; the managed
server brings its own Postgres.

## 4. Build and ship (recipe)

```powershell
langgraph build -t my-agent          # needs Docker; reads langgraph.json
```

Run the image with the model keys and `LANGSMITH_API_KEY` as host environment variables,
never baked into the image. `.env` stays on your machine (`.dockerignore` it if you add one).

## 5. Before real users

- [ ] Replace every placeholder in `AGENTS.md` ("Placeholders to replace, not ship").
- [ ] Gate every tool that writes, spends or sends in `agent/orchestration/approval.py`.
- [ ] Decide on `MEMORY_REQUIRE_APPROVAL`. Saved facts are read into the system prompt, so
      a poisoned `remember` is a prompt attack. `true` puts a human in front of it.
- [ ] Set `CONTEXT_WINDOW_TOKENS` to the real limit of the model you chose.
- [ ] Put a rate limit in front of the server. The template caps loops, not callers.
- [ ] Replace the eval cases and run `python -m evals` against the real model.
- [ ] Turn on tracing (`LANGSMITH_API_KEY`) so a bad run can be replayed.
