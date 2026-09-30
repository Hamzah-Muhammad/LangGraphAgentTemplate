"""
The agent, one folder per building block. README.md has the full tree.

    agent/
    ├── model/          BLOCK 1  MODEL           model.py
    ├── tools/          BLOCK 2  TOOLS           example_tool, memory_tools, files, skills, mcp
    ├── prompt/         BLOCK 3  SYSTEM PROMPT   system.md, loader, skills (+ skills/*/SKILL.md)
    ├── context/        BLOCK 4  CONTEXT WINDOW  compaction, offload
    ├── memory/         BLOCK 5  MEMORY          store
    ├── orchestration/  BLOCK 6  ORCHESTRATION   graph, reliability, approval, studio,
    │                                            patterns/, subagents/
    └── shared/         not a block              settings, runtime, schemas, usage

The blocks meet in one place: orchestration/graph.py (build_simple_agent).
"""
