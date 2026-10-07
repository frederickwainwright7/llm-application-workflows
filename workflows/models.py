"""Return the right LangChain chat model for a model ID on one OpenAI/Anthropic-compatible endpoint.

    OPENAI_API_KEY    key for the endpoint (sent as a Bearer token, or as x-api-key to the Anthropic client)
    OPENAI_BASE_URL   defaults to https://argolink.io/v1

Per model family:
- gpt-* and grok-*: ChatOpenAI on the Responses API. GPT-6 Sol needs it to call
  tools while reasoning; Chat Completions only allows its tools at effort "none".
- claude-*: ChatAnthropic on the root origin. The Anthropic SDK appends
  /v1/messages itself, so base_url must not end in /v1.
- everything else (deepseek-*, glm-*, kimi-*): ChatOpenAI on Chat Completions,
  where reasoning_effort is applied exactly as each model page documents.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://argolink.io/v1"


def family(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "grok-")):
        return "responses"
    return "chat_completions"


def chat_model(model: str, *, effort: str | None = None, base_url: str | None = None,
               api_key: str | None = None, **kwargs: Any) -> BaseChatModel:
    """effort maps to reasoning.effort, output_config.effort or reasoning_effort by family.
    Extra kwargs go to the LangChain class (http_client, timeout, max_retries, ...)."""
    base = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("set OPENAI_API_KEY to the key for your endpoint")

    kind = family(model)
    if kind == "anthropic":
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        if effort:
            kwargs["output_config"] = {"effort": effort}
        # max_tokens is required by the Messages API; temperature/top_p/top_k stay unset
        # because claude-sonnet-5 returns 400 for any non-default value.
        return ChatAnthropic(model=model, base_url=root, api_key=key,
                             max_tokens=kwargs.pop("max_tokens", 8192), **kwargs)
    if kind == "responses":
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        return ChatOpenAI(model=model, base_url=base, api_key=key, use_responses_api=True, **kwargs)
    if effort:
        kwargs["reasoning_effort"] = effort
    return ChatOpenAI(model=model, base_url=base, api_key=key, **kwargs)
