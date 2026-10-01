# agent/orchestration/studio.py  |  BLOCK 6 ORCHESTRATION
"""
Graph factories for langgraph.json (LangGraph Studio via `langgraph dev`, or Platform).
One entry per mode, so Studio shows each workflow as its own graph. The server supplies
persistence, so no checkpointer or store is passed here.
"""

from dotenv import load_dotenv

from agent.model.model import build_fallback_model, build_grader_model, build_model
from agent.orchestration.graph import build_graph


async def _make(mode: str):
    load_dotenv()
    return await build_graph(
        build_model(),
        mode=mode,
        fallback_model=build_fallback_model(),
        grader_model=build_grader_model(),
    )


async def simple():
    return await _make("simple")


async def planner():
    return await _make("planner")


async def router():
    return await _make("router")


async def evaluator():
    return await _make("evaluator")


async def supervisor():
    return await _make("supervisor")
