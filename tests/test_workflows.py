"""Offline tests: httpx mock transports stand in for the endpoint, so no key or network is needed.
They pin the exact URL and request body each LangChain class sends."""
import itertools
import json
import os
import sys

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflows"))
import models  # noqa: E402
from extract import ActionItems, extract, extractor  # noqa: E402
from summarize_long import chunks, summarize  # noqa: E402
from support_agent import build_agent  # noqa: E402
from triage_graph import Triage, build_graph  # noqa: E402


RESPONSE_IDS = itertools.count(1)


def responses_body(model, output, usage=(40, 10)):
    # unique ids matter: LangGraph merges messages that share an id
    return {"id": f"resp_{next(RESPONSE_IDS)}", "object": "response", "created_at": 0, "status": "completed", "model": model, "output": output,
            "usage": {"input_tokens": usage[0], "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": usage[1], "output_tokens_details": {"reasoning_tokens": 0},
                      "total_tokens": sum(usage)}}


def text_item(text):
    return {"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


@pytest.fixture
def endpoint():
    """Records requests; reply(request_json) -> response json is set per test."""
    state = {"seen": [], "reply": None}

    def handler(request):
        body = json.loads(request.content)
        state["seen"].append((str(request.url), dict(request.headers), body))
        return httpx.Response(200, json=state["reply"](body))

    state["client"] = httpx.Client(transport=httpx.MockTransport(handler))
    return state


def test_each_family_gets_the_right_class_and_base_url():
    gpt = models.chat_model("gpt-6-sol", api_key="test")
    claude = models.chat_model("claude-sonnet-5", api_key="test")
    glm = models.chat_model("glm-5.3", effort="low", api_key="test")
    assert gpt.use_responses_api is True and gpt.openai_api_base == "https://argolink.io/v1"
    assert type(claude).__name__ == "ChatAnthropic" and claude.anthropic_api_url == "https://argolink.io"
    assert glm.use_responses_api is None and glm.reasoning_effort == "low"


def test_gpt_goes_to_responses_with_reasoning_effort(endpoint):
    endpoint["reply"] = lambda body: responses_body(body["model"], [text_item("pong")])
    reply = models.chat_model("gpt-6-sol", effort="high", api_key="test", http_client=endpoint["client"]).invoke("ping")
    url, headers, body = endpoint["seen"][-1]
    assert url == "https://argolink.io/v1/responses"
    assert headers["authorization"] == "Bearer test"
    assert body["reasoning"] == {"effort": "high"}
    assert reply.text == "pong"


def test_glm_goes_to_chat_completions_with_reasoning_effort(endpoint):
    endpoint["reply"] = lambda body: {
        "id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "pong"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}
    models.chat_model("glm-5.3", effort="max", api_key="test", http_client=endpoint["client"]).invoke("ping")
    url, _, body = endpoint["seen"][-1]
    assert url == "https://argolink.io/v1/chat/completions"
    assert body["reasoning_effort"] == "max" and "reasoning" not in body


def test_claude_uses_root_origin_effort_and_no_temperature(monkeypatch):
    import langchain_anthropic.chat_models as anthropic_module

    seen = []

    def handler(request):
        seen.append((str(request.url), dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "pong"}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 8, "output_tokens": 1}})

    monkeypatch.setattr(anthropic_module, "_get_default_httpx_client",
                        lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), base_url=kw["base_url"]))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://argolink.io/v1")
    models.chat_model("claude-sonnet-5", effort="medium", api_key="test").invoke("ping")
    url, headers, body = seen[-1]
    assert url == "https://argolink.io/v1/messages"  # not /v1/v1/messages
    assert headers["x-api-key"] == "test"
    assert body["output_config"] == {"effort": "medium"}
    assert not {"temperature", "top_p", "top_k"} & body.keys()


def test_extraction_sends_a_strict_json_schema(endpoint):
    items = {"items": [{"owner": "Priya", "task": "Send the revised SOW", "due": "2026-10-14"}]}
    endpoint["reply"] = lambda body: responses_body(body["model"], [text_item(json.dumps(items))])
    result = extract("Priya sends the SOW by Oct 14.", extractor(api_key="test", http_client=endpoint["client"]))
    fmt = endpoint["seen"][-1][2]["text"]["format"]
    assert (fmt["type"], fmt["strict"], fmt["name"]) == ("json_schema", True, "ActionItems")
    assert isinstance(result, ActionItems) and result.items[0].owner == "Priya"


def test_extraction_retries_an_unparseable_reply():
    replies = iter([{"parsed": None, "parsing_error": "bad json", "raw": None},
                    {"parsed": ActionItems(items=[]), "parsing_error": None, "raw": None}])
    assert extract("nothing to do", RunnableLambda(lambda _: next(replies))).items == []


def test_agent_calls_the_tool_then_answers(endpoint):
    def reply(body):
        if any(item.get("type") == "function_call_output" for item in body["input"]):
            return responses_body(body["model"], [text_item("A-1001 shipped on Oct 3 and is returnable.")])
        return responses_body(body["model"], [{
            "type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup_order",
            "arguments": json.dumps({"order_id": "A-1001"}), "status": "completed"}])

    endpoint["reply"] = reply
    agent = build_agent(models.chat_model("gpt-6-sol", api_key="test", http_client=endpoint["client"]))
    result = agent.invoke({"messages": [{"role": "user", "content": "Where is A-1001?"}]})
    assert len(endpoint["seen"]) == 2
    tool_output = next(i for i in endpoint["seen"][1][2]["input"] if i.get("type") == "function_call_output")
    assert json.loads(tool_output["output"])["status"] == "shipped"
    assert result["messages"][-1].text.startswith("A-1001 shipped")


@pytest.mark.parametrize("triage,route", [
    (Triage(category="bug", urgent=False), "escalate"),
    (Triage(category="billing", urgent=True), "escalate"),
    (Triage(category="how-to", urgent=False), "reply"),
])
def test_triage_graph_routes_by_classification(triage, route):
    graph = build_graph(classifier=RunnableLambda(lambda _: triage),
                        replier=RunnableLambda(lambda _: AIMessage("reply draft")),
                        escalator=RunnableLambda(lambda _: AIMessage("handoff draft")))
    state = graph.invoke({"text": "ticket"})
    assert state["route"] == route


def test_chunks_respect_the_size_limit():
    text = "\n\n".join("x" * 1_000 for _ in range(100))
    parts = chunks(text, size=10_000)
    assert len(parts) >= 10 and all(len(p) <= 10_000 for p in parts)
    assert sum(p.count("x") for p in parts) == 100_000


def test_map_reduce_merges_every_partial():
    mapper = RunnableLambda(lambda prompt: AIMessage(f"- part of {len(prompt)} chars"))
    seen = {}
    reducer = RunnableLambda(lambda prompt: seen.setdefault("prompt", prompt) and AIMessage("final"))
    assert summarize("a\n\n" * 30_000, mapper=mapper, reducer=reducer) == "final"
    assert seen["prompt"].count("Part ") == len(chunks("a\n\n" * 30_000))
