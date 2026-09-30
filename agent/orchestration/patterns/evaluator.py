# agent/orchestration/patterns/evaluator.py  |  BLOCK 6 ORCHESTRATION
"""
EVALUATOR-OPTIMIZER

    start -> generate -> grade -> (pass) finish
                               -> (fail, rounds left) generate again with the feedback
                               -> (fail, out of rounds, or grader broken) human -> finish

Rules from the research, all enforced here:
  - the generator is the full simple agent, so it can use tools, memory and skills while
    drafting, and it sees the whole conversation (follow-ups work)
  - the grader is a SEPARATE model call with its own prompt, ideally a different model
    (GRADER_* in .env). Agents grade their own work too kindly.
  - the verdict is structured output (Verdict: passed + feedback)
  - rounds are capped (MAX_EVAL_ROUNDS, default 3); after that a human decides
  - a grader that returns malformed output escalates to the human at once, instead of
    crashing the run or passing an unchecked draft
  - feedback must be specific, or revisions go nowhere
  - per-request state resets at the start of every turn

Human escalation: the graph pauses with the draft and last feedback. Resume with
{"draft": "<edited text>"} to replace it, or anything else to accept it as is.
"""

from typing import Annotated, Any, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from typing_extensions import TypedDict

from agent.context.history import latest_request
from agent.model.structured import structured
from agent.shared.runtime import Context
from agent.shared.schemas import Verdict

GRADER_PROMPT = """You grade a draft written by someone else against the user's request.
Pass it only if it fully answers the request, is correct, and follows any stated format.
If it fails, say exactly what is missing or wrong and how to fix it. Be specific."""

REVISE_PROMPT = """Your previous draft:
{draft}

A reviewer rejected it with this feedback:
{feedback}

Write a new draft that fixes every point. Output only the draft."""


class EvalState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: NotRequired[str]
    feedback: NotRequired[str]
    passed: NotRequired[bool]
    grader_failed: NotRequired[bool]
    rounds: NotRequired[int]


def build_evaluator_graph(
    generator: Any,
    grader: BaseChatModel,
    *,
    max_rounds: int = 3,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    """generator: a compiled agent taking {"messages": [...]} (normally build_simple_agent).
    grader: a chat model; only its structured Verdict is used."""
    judge = structured(grader, Verdict)

    def start(state: EvalState) -> dict:
        # Each user turn is a new job. Without this reset, turn 2 would "revise" turn 1's
        # draft with turn 1's rounds already spent.
        return {"draft": "", "feedback": "", "passed": False, "grader_failed": False,
                "rounds": 0}

    async def generate(state: EvalState, runtime: Runtime[Context]) -> dict:
        messages = list(state["messages"])  # whole conversation, so follow-ups resolve
        if state.get("draft"):
            revise = REVISE_PROMPT.format(draft=state["draft"], feedback=state["feedback"])
            messages.append(HumanMessage(revise))
        out = await generator.ainvoke({"messages": messages}, context=runtime.context)
        return {"draft": str(out["messages"][-1].content), "rounds": state.get("rounds", 0) + 1}

    async def grade(state: EvalState) -> dict:
        request = latest_request(state["messages"])
        try:
            verdict: Verdict = await judge.ainvoke(
                [SystemMessage(GRADER_PROMPT),
                 HumanMessage(f"Request:\n{request}\n\nDraft:\n{state['draft']}")]
            )
        except Exception as error:  # malformed verdict: a human decides, never auto-pass
            return {"passed": False, "grader_failed": True,
                    "feedback": f"The grader failed ({type(error).__name__}); review manually."}
        return {"passed": verdict.passed, "feedback": verdict.feedback}

    def after_grade(state: EvalState) -> str:
        if state["passed"]:
            return "finish"
        if state.get("grader_failed") or state["rounds"] >= max_rounds:
            return "escalate"
        return "generate"

    def escalate(state: EvalState) -> dict:
        reason = ("The grader could not produce a verdict." if state.get("grader_failed")
                  else f"Draft failed review {state['rounds']} times.")
        decision = interrupt(
            {
                "type": "evaluator_escalation",
                "message": f"{reason} Edit or accept it.",
                "draft": state["draft"],
                "feedback": state["feedback"],
            }
        )
        if isinstance(decision, dict) and decision.get("draft"):
            return {"draft": decision["draft"]}
        return {}

    def finish(state: EvalState) -> dict:
        return {"messages": [AIMessage(content=state["draft"])]}

    builder = StateGraph(EvalState, context_schema=Context)
    builder.add_node("start", start)
    builder.add_node("generate", generate)
    builder.add_node("grade", grade)
    builder.add_node("escalate", escalate)
    builder.add_node("finish", finish)
    builder.add_edge(START, "start")
    builder.add_edge("start", "generate")
    builder.add_edge("generate", "grade")
    builder.add_conditional_edges("grade", after_grade, ["finish", "generate", "escalate"])
    builder.add_edge("escalate", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer, store=store, name="evaluator")
