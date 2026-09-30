"""
WORKFLOW PATTERNS

Each file is one pattern from docs/PATTERNS.md, built as a LangGraph StateGraph from the
same blocks as the simple agent. Copy the one closest to your flow and edit it.

  planner.py     orchestrator-workers: plan, parallel workers, synthesize
  router.py      classify (rules first, model second), send to one specialist path
  evaluator.py   generator + separate grader, capped rounds, human escalation
  supervisor.py  central coordinator delegating to specialist agents via Command(goto)
"""
