"""
CLI entry point. Wires env -> blocks -> graph, then streams a chat loop.

    python run.py                          simple agent, thread "default"
    python run.py --mode planner           plan -> parallel workers -> synthesize
    python run.py --mode router            rules/model pick simple or planner per request
    python run.py --mode evaluator         draft -> separate grader -> revise (max 3) -> human
    python run.py --mode supervisor        coordinator + researcher + writer
    python run.py --thread demo --user bob separate memory thread, different user namespace
    python run.py --once "hello"           single turn, then exit

Time travel (debugging):
    python run.py --thread demo --history           list checkpoints, newest first
    python run.py --thread demo --replay <id>       re-run from that checkpoint

Streams tokens as they arrive and prints tool calls as they happen.
Approval: a gated tool pauses the run. Answer y (approve), n (reject) or e (edit args as
JSON). Several branches can pause at once (planner); each is answered, and the resume is
one {interrupt_id: value} map, which is how LangGraph pairs them.

Uses the stable `astream(stream_mode=[...])` API. LangGraph's `stream_events(v3)` is
still marked experimental in 1.2.
"""

import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# LangSmith tracing turns on only if a key is present.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from agent.graph import MODES, build_graph  # noqa: E402
from agent.memory import open_memory  # noqa: E402
from agent.model import build_fallback_model, build_grader_model, build_model  # noqa: E402
from agent.runtime import Context  # noqa: E402
from agent.usage import usage_from_messages  # noqa: E402


async def stream_run(graph, payload, config, context) -> list:
    """Stream one graph run. Prints tokens and tool events. Returns pending interrupts."""
    interrupts = []
    async for mode, chunk in graph.astream(
        payload, config=config, context=context, stream_mode=["messages", "updates"],
        subgraphs=False,
    ):
        if mode == "messages":
            msg, _meta = chunk
            if msg.type == "AIMessageChunk" and isinstance(msg.content, str) and msg.content:
                print(msg.content, end="", flush=True)
        elif mode == "updates":
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupts.extend(update)
                    continue
                for m in (update or {}).get("messages", []) or []:
                    if getattr(m, "type", "") == "tool":
                        print(f"\n[tool:{m.name}] {str(m.content)[:200]}")
    print()
    return interrupts


def ask_resume(interrupt):
    """Turn one interrupt into its resume value by asking the human."""
    value = interrupt.value
    if isinstance(value, dict) and value.get("type") == "evaluator_escalation":
        print(f"\n[review] {value['message']}\nfeedback: {value['feedback']}\n\n{value['draft']}")
        edited = input("\nenter to accept, or type a replacement draft: ").strip()
        return {"draft": edited} if edited else {}

    requests = value.get("action_requests", [value]) if isinstance(value, dict) else [value]
    decisions = []
    for req in requests:
        print(f"\n[approval] {req.get('name')} {json.dumps(req.get('args', {}))}")
        choice = input("approve (y) / reject (n) / edit (e): ").strip().lower()
        if choice == "y":
            decisions.append({"type": "approve"})
        elif choice == "e":
            new_args = json.loads(input("new args as JSON: "))
            decisions.append(
                {"type": "edit", "edited_action": {"name": req["name"], "args": new_args}}
            )
        else:
            decisions.append({"type": "reject", "message": "Rejected by user."})
    return {"decisions": decisions}


async def drive(graph, payload, config, resume_config, context) -> None:
    """Run until finished, answering every interrupt along the way."""
    print("\nagent> ", end="", flush=True)
    interrupts = await stream_run(graph, payload, config, context)
    while interrupts:
        resume_map = {i.id: ask_resume(i) for i in interrupts}
        print("\nagent> ", end="", flush=True)
        interrupts = await stream_run(graph, Command(resume=resume_map), resume_config, context)


async def turn(graph, config, context, text: str) -> None:
    before = len((await graph.aget_state(config)).values.get("messages", []))
    payload = {"messages": [{"role": "user", "content": text}]}
    await drive(graph, payload, config, config, context)
    messages = (await graph.aget_state(config)).values.get("messages", [])
    print(f"[{usage_from_messages(messages[before:])}]")


async def print_history(graph, config) -> None:
    print("step  checkpoint_id                          next          last message")
    async for snap in graph.aget_state_history(config, limit=50):
        cid = snap.config["configurable"]["checkpoint_id"]
        msgs = snap.values.get("messages", [])
        last = f"{msgs[-1].type}: {str(msgs[-1].content)[:50]!r}" if msgs else ""
        nxt = ",".join(snap.next) or "-"
        print(f"{snap.metadata.get('step', ''):>4}  {cid}  {nxt:<12}  {last}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="simple")
    parser.add_argument("--thread", default="default", help="memory thread id")
    parser.add_argument("--user", default=os.getenv("DEFAULT_USER_ID", "anonymous"))
    parser.add_argument("--once", help="run a single user message and exit")
    parser.add_argument("--history", action="store_true", help="list checkpoints and exit")
    parser.add_argument("--replay", metavar="CHECKPOINT_ID", help="re-run from a checkpoint")
    args = parser.parse_args()

    async with open_memory() as (checkpointer, store):
        graph = await build_graph(
            build_model(),
            mode=args.mode,
            fallback_model=build_fallback_model(),
            grader_model=build_grader_model(),
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": args.thread}}
        context = Context(user_id=args.user)

        if args.history:
            await print_history(graph, config)
            return
        if args.replay:
            at = {"configurable": {"thread_id": args.thread, "checkpoint_id": args.replay}}
            await drive(graph, None, at, config, context)  # None = continue from `at`
            return
        if args.once:
            await turn(graph, config, context, args.once)
            return

        print(f"mode={args.mode} thread={args.thread} user={args.user}  (ctrl-c to quit)")
        while True:
            try:
                text = input("\nyou> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if text:
                await turn(graph, config, context, text)


if __name__ == "__main__":
    asyncio.run(main())
