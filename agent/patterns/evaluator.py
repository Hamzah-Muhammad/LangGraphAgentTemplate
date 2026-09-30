"""
EVALUATOR-OPTIMIZER

    generate -> grade -> (pass) finish
                      -> (fail, rounds left) generate again with the feedback
                      -> (fail, out of rounds) escalate to a human -> finish

Rules from the research, all enforced here:
  - the grader is a SEPARATE model call with its own prompt, ideally a different model
    (GRADER_* in .env). Agents grade their own work too kindly.
  - the verdict is structured output (Verdict: passed + feedback)
  - rounds are capped (MAX_EVAL_ROUNDS, default 3); after that a human decides
  - feedback must be specific, or revisions go nowhere

Human escalation: the graph pauses with the draft and last feedback. Resume with
{"draft": "<edited text>"} to replace it, or anything else to accept it as is.
"""

from typing import Annotated, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from typing_extensions import TypedDict

from agent.graph import load_system_prompt
from agent.runtime import Context
from agent.schemas import Verdict

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
    rounds: NotRequired[int]


def build_evaluator_graph(
    generator: BaseChatModel,
    grader: BaseChatModel,
    *,
    max_rounds: int = 3,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
):
    judge = grader.with_structured_output(Verdict)

    def request_of(state: EvalState) -> str:
        humans = [m for m in state["messages"] if m.type == "human"]
        return str(humans[-1].content) if humans else ""

    async def generate(state: EvalState) -> dict:
        request = request_of(state)
        prompt = [SystemMessage(load_system_prompt()), HumanMessage(request)]
        if state.get("draft"):
            prompt.append(
                HumanMessage(REVISE_PROMPT.format(draft=state["draft"], feedback=state["feedback"]))
            )
        reply = await generator.ainvoke(prompt)
        return {"draft": str(reply.content), "rounds": state.get("rounds", 0) + 1}

    async def grade(state: EvalState) -> dict:
        verdict: Verdict = await judge.ainvoke(
            [
                SystemMessage(GRADER_PROMPT),
                HumanMessage(f"Request:\n{request_of(state)}\n\nDraft:\n{state['draft']}"),
            ]
        )
        return {"passed": verdict.passed, "feedback": verdict.feedback}

    def after_grade(state: EvalState) -> str:
        if state["passed"]:
            return "finish"
        return "escalate" if state["rounds"] >= max_rounds else "generate"

    def escalate(state: EvalState) -> dict:
        decision = interrupt(
            {
                "type": "evaluator_escalation",
                "message": f"Draft failed review {state['rounds']} times. Edit or accept it.",
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
    builder.add_node("generate", generate)
    builder.add_node("grade", grade)
    builder.add_node("escalate", escalate)
    builder.add_node("finish", finish)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "grade")
    builder.add_conditional_edges("grade", after_grade, ["finish", "generate", "escalate"])
    builder.add_edge("escalate", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer, store=store, name="evaluator")
