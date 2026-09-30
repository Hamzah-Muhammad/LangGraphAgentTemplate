# Workflow patterns: which one, when, and what it costs

Start with the simplest thing that works. Every step up this list adds model calls,
latency and new ways to fail. Move up only when an eval shows the simpler mode losing.

| Mode | Shape | Use when | Model calls per request | Main failure mode | File |
|---|---|---|---|---|---|
| `simple` | model ↔ tools loop | Almost always. One goal, tools as needed. | 1 + one per tool round | Loops or wanders on long tasks | `agent/graph.py` |
| `router` | classify → one path | Requests fall into clearly different kinds that need different handling | +0 if a rule matches, +1 if the model classifies | Misrouting on ambiguous input | `agent/patterns/router.py` |
| `planner` | plan → N parallel workers → synthesize | Breadth-first work: several independent directions | 2 + N workers (each a full agent). Roughly 5-15x `simple` | Overlapping or vague sub-tasks, runaway fan-out | `agent/patterns/planner.py` |
| `evaluator` | draft → grade → revise, capped | Output quality has clear, checkable criteria | 2 per round, max 3 rounds | Vague grading criteria give vague revisions | `agent/patterns/evaluator.py` |
| `supervisor` | coordinator → specialist → coordinator … | Distinct roles with different prompts or tools (researcher, writer, reviewer) | 1 per hop + each specialist's own calls | Ping-pong between specialists | `agent/patterns/supervisor.py` |

Patterns compose. The default `router` already sends breadth-first requests to `planner`
and everything else to `simple`. A supervisor's specialist can itself be a planner.

## The rules each pattern enforces in code

Asking a model to behave is a suggestion. These are enforced.

**Every mode**
- Model calls per run are capped (`agent/reliability.py`, 25).
- Transient model errors retry, then fall back to a second model if one is configured.
- A failing tool retries once, then its error reaches the model as text, not as a crash.
- Old tool results are cleared at 50% of the context window. History is summarized at 60%.
  Compaction happens on your terms, before the model is under pressure.
- Any single tool result over `OFFLOAD_CHARS` is saved to a file. The model sees a preview
  and pages through the rest with `read_file`.
- Gated tools pause for a human before they run.

**Router**
- Regex rules run before the model. A rule match costs zero model calls.
- The model classifier answers with a `Literal` of real route names, so it cannot invent one.
- If the classifier fails, a default route runs.

**Planner**
- The plan is structured output you can log.
- Worker count is capped in code (`MAX_WORKERS`), not only in the prompt.
- The planner prompt carries one example of a good decomposition and says to return one
  step for a focused question.
- Workers get a word budget so the synthesizer receives distilled results.
- The worker node has a retry policy.

**Evaluator**
- The grader is a separate call with its own prompt, and can be a separate model (`GRADER_*`).
  Models grade their own work too kindly.
- The verdict is structured: `passed` plus specific `feedback`.
- After `MAX_EVAL_ROUNDS` failures a human edits or accepts the draft.

**Supervisor**
- The supervisor never does the work. It picks one specialist per turn or finishes.
- Specialists see only their instruction, not the whole conversation.
- Control always returns to the supervisor, so every routing decision is on one audit trail.
- Delegations are capped (`MAX_SUPERVISOR_HOPS`).

## Supervisor or swarm?

Supervisor is the default in 2026. Claude Code subagents, the OpenAI Agents SDK and
LangGraph all converge on it. It costs about twice the model calls of a swarm and buys
more accurate routing and one audit trail.

A swarm (agents hand off to each other with `Command(goto=...)`, no central coordinator)
fits when the flow is unpredictable and latency matters more than auditability, such as a
support bot passing a customer between departments. To build one, copy `supervisor.py`
and give each specialist the handoff decision instead of the supervisor.

## Harness features that are not patterns

| Feature | Where | Why |
|---|---|---|
| Skills (`skills/<name>/SKILL.md`) | `agent/skills.py`, `tools/skills.py` | Reusable procedures. Only a one-line index sits in the prompt; the body loads on demand. |
| Todo list (`AGENT_TODOS=true`) | `agent/context.py` | A `write_todos` tool that keeps the plan in recent attention on long tasks. |
| Long-term memory | `tools/memory_tools.py` | `remember` / `recall`, namespaced per user, survives restarts. |
| Time travel | `run.py --history`, `--replay` | Re-run a thread from any checkpoint to debug a bad step. |
| Trajectory evals | `evals/` | Check which tools ran and in what order, not only the final text. |

## Long tasks that outgrow one context

For work longer than any context window, run the agent in a loop where each iteration
starts with a fresh context and reads its state from files the last iteration wrote
(a plan file, a progress log). The offload directory and `read_file` are the building
blocks. This is left as a pattern to copy rather than a mode, because the state files are
specific to each task.

## Sources

- Anthropic, Building Effective Agents (workflow taxonomy, "start simple")
- Anthropic, multi-agent research system (orchestrator-workers gains and ~15x token cost)
- LangGraph docs: interrupts, subgraphs, time travel, durable execution
- LangChain Deep Agents (todos, file offload, subagents as context isolation)
- 2026 harness-engineering and agent-evaluation write-ups (compaction thresholds,
  separate graders, trajectory evals, supervisor as default topology)
