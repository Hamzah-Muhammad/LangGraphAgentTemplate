"""
CLI entry point. Wires env -> blocks -> graph, then streams a chat loop.

    python run.py                          interactive, simple agent, thread "default"
    python run.py --mode planner           plan -> parallel workers -> synthesize
    python run.py --thread demo --user bob separate memory thread, different user namespace
    python run.py --once "hello"           single turn, then exit

Streams tokens as they arrive and prints tool calls as they happen.
Approval flow: when a gated tool is called the graph pauses. You are shown the call and
type  y  (approve),  n  (reject), or  e  (edit args as JSON). The graph resumes.
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

from agent.graph import build_graph  # noqa: E402
from agent.memory import open_memory  # noqa: E402
from agent.model import build_fallback_model, build_model  # noqa: E402
from agent.runtime import Context  # noqa: E402
from agent.usage import usage_from_messages  # noqa: E402


async def stream_run(graph, payload, config, context) -> list:
    """Stream one graph run. Prints tokens and tool events. Returns pending interrupts."""
    interrupts = []
    async for mode, chunk in graph.astream(
        payload, config=config, context=context, stream_mode=["messages", "updates"]
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


def ask_decisions(interrupt) -> list[dict]:
    requests = interrupt.value.get("action_requests", [interrupt.value])
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
    return decisions


async def turn(graph, config, context, text: str) -> None:
    before = len((await graph.aget_state(config)).values.get("messages", []))
    payload = {"messages": [{"role": "user", "content": text}]}
    print("\nagent> ", end="", flush=True)
    interrupts = await stream_run(graph, payload, config, context)
    while interrupts:
        decisions = ask_decisions(interrupts[0])
        print("\nagent> ", end="", flush=True)
        interrupts = await stream_run(
            graph, Command(resume={"decisions": decisions}), config, context
        )
    messages = (await graph.aget_state(config)).values.get("messages", [])
    print(f"[{usage_from_messages(messages[before:])}]")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simple", "planner"], default="simple")
    parser.add_argument("--thread", default="default", help="memory thread id")
    parser.add_argument("--user", default=os.getenv("DEFAULT_USER_ID", "anonymous"))
    parser.add_argument("--once", help="run a single user message and exit")
    args = parser.parse_args()

    async with open_memory() as (checkpointer, store):
        graph = await build_graph(
            build_model(),
            mode=args.mode,
            fallback_model=build_fallback_model(),
            checkpointer=checkpointer,
            store=store,
        )
        config = {"configurable": {"thread_id": args.thread}}
        context = Context(user_id=args.user)

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
