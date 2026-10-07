"""Workflow 3: a tool-calling support agent (LangChain create_agent, which runs on LangGraph).

GPT-6 Sol calls tools while reasoning only on the Responses API, which
chat_model() selects for gpt-* models.

    python workflows/support_agent.py "Where is order A-1001, and can I still return it?"
"""
import json
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool

from models import chat_model

ORDERS = {"A-1001": {"status": "shipped", "shipped_on": "2026-10-03", "days_since_delivery": 2}}


@tool
def lookup_order(order_id: str) -> str:
    """Shipping status of an order by its ID, e.g. A-1001."""
    return json.dumps(ORDERS.get(order_id, {"error": "unknown order"}))


@tool
def return_policy(days_since_delivery: int) -> str:
    """Whether an order can still be returned, given days since delivery."""
    return "returnable" if days_since_delivery <= 30 else "outside the 30-day window"


def build_agent(model=None):
    return create_agent(
        model or chat_model("gpt-6-sol", effort="medium"),
        tools=[lookup_order, return_policy],
        system_prompt="You are a support agent. Use the tools; never guess an order's status.",
    )


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Where is order A-1001, and can I still return it?"
    result = build_agent().invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].text)
