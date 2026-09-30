"""python -m evals : run every case against the real model from .env."""

import asyncio
import os
import sys

from dotenv import load_dotenv

from agent.model.model import build_fallback_model, build_model
from agent.orchestration.graph import build_graph
from evals.runner import fresh_checkpointer, load_cases, run_all


async def main() -> int:
    load_dotenv()
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    model, fallback = build_model(), build_fallback_model()

    async def factory(_case):
        return await build_graph(model, fallback_model=fallback, checkpointer=fresh_checkpointer())

    results = await run_all(factory, load_cases())
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"{mark}  {r.case.id}  tools={r.tools}  {'; '.join(r.reasons)}")
        if not r.passed:
            print(f"      output: {r.output[:200]!r}")
    failed = sum(not r.passed for r in results)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
