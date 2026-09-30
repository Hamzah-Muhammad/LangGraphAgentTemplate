"""
BLOCK 6: ORCHESTRATION. What runs next. Zero model reasoning.

    graph.py        build_simple_agent (assembles all blocks + middleware), build_graph (modes)
    reliability.py  call cap, retries, fallback, tool-error recovery
    approval.py     human gate: tools listed here pause before running
    studio.py       one graph factory per mode, for langgraph.json
    patterns/       planner, router, evaluator, supervisor
    subagents/      specialist (subagent as a tool), team (supervisor's specialists)
"""
