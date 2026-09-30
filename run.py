"""
CLI entry point. Wires env -> blocks -> graph, then streams a chat loop.

    python run.py                      interactive, thread "default"
    python run.py --thread demo        separate memory thread
    python run.py --once "hello"       single turn, then exit

Approval flow: when a gated tool is called the graph pauses. You are shown the call and
type  y  (approve),  n  (reject), or  e  (edit args as JSON). The graph resumes.
"""

import argparse
import json
import os

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# LangSmith tracing turns on only if a key is present.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from agent.graph import build_graph            # noqa: E402
from agent.memory import build_checkpointer, build_store  # noqa: E402
from agent.model import build_model            # noqa: E402


def handle_interrupts(graph, config, result):
    """Loop until no interrupt is pending, asking the human each time."""
    while "__interrupt__" in result:
        interrupt = result["__interrupt__"][0]
        requests = interrupt.value.get("action_requests", [interrupt.value])
        decisions = []
        for req in requests:
            print(f"\n[approval] {req.get('name')} {json.dumps(req.get('args', {}))}")
            choice = input("approve (y) / reject (n) / edit (e): ").strip().lower()
            if choice == "y":
                decisions.append({"type": "approve"})
            elif choice == "e":
                new_args = json.loads(input("new args as JSON: "))
                decisions.append({"type": "edit", "edited_action": {"name": req["name"], "args": new_args}})
            else:
                decisions.append({"type": "reject", "message": "Rejected by user."})
        result = graph.invoke(Command(resume={"decisions": decisions}), config=config)
    return result


def turn(graph, config, text: str) -> str:
    result = graph.invoke({"messages": [{"role": "user", "content": text}]}, config=config)
    result = handle_interrupts(graph, config, result)
    return result["messages"][-1].content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", default="default", help="memory thread id")
    parser.add_argument("--once", help="run a single user message and exit")
    args = parser.parse_args()

    graph = build_graph(build_model(), build_checkpointer(), build_store())
    config = {"configurable": {"thread_id": args.thread}}

    if args.once:
        print(turn(graph, config, args.once))
        return

    print(f"thread={args.thread}  (ctrl-c to quit)")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if text:
            print(f"\nagent> {turn(graph, config, text)}")


if __name__ == "__main__":
    main()
