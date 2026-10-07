"""Workflow 4: a LangGraph triage graph that spends money only where it matters.

    classify (cheap model, structured output)
        bug or urgent  ->  escalate (engineering handoff, stronger model)
        otherwise      ->  reply    (customer answer)

    python workflows/triage_graph.py "The invoice PDF export crashes every time since this morning"
"""
import sys
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from models import chat_model


class Triage(BaseModel):
    category: Literal["billing", "bug", "how-to", "account", "other"]
    urgent: bool


class Ticket(TypedDict, total=False):
    text: str
    triage: Triage
    draft: str
    route: str


def build_graph(classifier=None, replier=None, escalator=None):
    """Each model is injectable, so the graph can be tested without a network."""
    classifier = classifier or chat_model("gpt-5.6-luna").with_structured_output(Triage, method="json_schema")
    replier = replier or chat_model("gpt-5.6-terra")
    escalator = escalator or chat_model("claude-sonnet-5", effort="medium")

    def classify(state: Ticket) -> Ticket:
        return {"triage": classifier.invoke(f"Classify this support ticket:\n{state['text']}")}

    def reply(state: Ticket) -> Ticket:
        answer = replier.invoke(f"Write a short, friendly reply to this customer:\n{state['text']}")
        return {"draft": answer.text, "route": "reply"}

    def escalate(state: Ticket) -> Ticket:
        handoff = escalator.invoke(
            "Write an engineering handoff: symptom, likely component, what to check first.\n"
            f"Ticket ({state['triage'].category}, urgent={state['triage'].urgent}):\n{state['text']}")
        return {"draft": handoff.text, "route": "escalate"}

    def pick(state: Ticket) -> str:
        triage = state["triage"]
        return "escalate" if triage.category == "bug" or triage.urgent else "reply"

    graph = StateGraph(Ticket)
    graph.add_node("classify", classify)
    graph.add_node("reply", reply)
    graph.add_node("escalate", escalate)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", pick, ["reply", "escalate"])
    graph.add_edge("reply", END)
    graph.add_edge("escalate", END)
    return graph.compile()


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "How do I add a second admin to my workspace?"
    state = build_graph().invoke({"text": text})
    print(f"[{state['route']}] {state['triage'].model_dump()}\n\n{state['draft']}")
