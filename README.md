# LLM application workflows with LangChain and LangGraph

Five production-shaped workflows — streaming chat, structured extraction, a tool-calling agent, a LangGraph triage graph and map-reduce summarization — for LangChain 1.x, running GPT, Claude, DeepSeek and GLM models through one OpenAI- and Anthropic-compatible endpoint. Unofficial; not affiliated with LangChain.

[Workflows](#the-five-workflows) · [Model factory](#one-factory-for-every-model-family) · [Structured output](#structured-output-what-each-class-sends) · [Models and prices](#models-and-prices) · [Costs](#what-each-workflow-costs) · [FAQ](#faq)

<sub>Last checked: 2026-10-07, with langchain 1.4.3, langchain-core 1.6.7, langchain-openai 1.6.7, langchain-anthropic 1.7.5 and langgraph 1.2.14. Every request shape below is pinned by an offline test.</sub>

## The five workflows

| # | Workflow | LangChain pieces | Default model(s) | File |
|---|---|---|---|---|
| 1 | Streaming chat with token counts | `stream()`, `usage_metadata` | `glm-5.3-flash` | [chat_stream.py](./workflows/chat_stream.py) |
| 2 | Meeting notes → validated action items | `with_structured_output(..., method="json_schema", include_raw=True)` | `gpt-6-sol` | [extract.py](./workflows/extract.py) |
| 3 | Support agent with tools | `create_agent`, `@tool` | `gpt-6-sol` | [support_agent.py](./workflows/support_agent.py) |
| 4 | Ticket triage graph | LangGraph `StateGraph`, conditional edges | `gpt-5.6-luna` → `gpt-5.6-terra` or `claude-sonnet-5` | [triage_graph.py](./workflows/triage_graph.py) |
| 5 | Map-reduce summary of a long document | `batch(..., max_concurrency=4)` | `deepseek-v4.1-flash` → `gpt-6-sol` | [summarize_long.py](./workflows/summarize_long.py) |

## Setup

```bash
git clone https://github.com/frederickwainwright7/llm-application-workflows
cd llm-application-workflows
pip install -r requirements.txt            # Python 3.10+
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=https://argolink.io/v1

python workflows/extract.py
python workflows/support_agent.py "Where is order A-1001, and can I still return it?"
python workflows/triage_graph.py "The invoice PDF export crashes every time since this morning"
```

The defaults point at [ArgoLink](https://argolink.io/gh-llm-application-workflows), which serves every model used here under one key on all three protocols (Responses, Chat Completions, Anthropic Messages). Any endpoint that speaks those protocols works; change the model IDs to ones it serves.

## One factory for every model family

[`workflows/models.py`](./workflows/models.py) returns the right LangChain class for a model ID, so the workflows never hard-code a vendor:

```python
from models import chat_model

sol = chat_model("gpt-6-sol", effort="medium")        # ChatOpenAI, Responses API
claude = chat_model("claude-sonnet-5", effort="high") # ChatAnthropic, Anthropic Messages
glm = chat_model("glm-5.3", effort="low")             # ChatOpenAI, Chat Completions
```

| Model IDs | LangChain class | Endpoint it calls | `effort=` becomes | Why |
|---|---|---|---|---|
| `gpt-*`, `grok-*` | `ChatOpenAI(use_responses_api=True)` | `/v1/responses` | `reasoning={"effort": ...}` | native format; GPT-6 Sol calls tools while reasoning only here |
| `claude-*` | `ChatAnthropic(base_url="https://argolink.io")` | `/v1/messages` | `output_config={"effort": ...}` | native format; effort, thinking and prompt caching arrive as sent |
| `deepseek-*`, `glm-*`, `kimi-*` | `ChatOpenAI()` | `/v1/chat/completions` | `reasoning_effort=...` | native format; thinking controls behave exactly as documented |

Three things we confirmed while pinning the requests:

1. **`ChatAnthropic` needs the root origin.** The Anthropic SDK appends `/v1/messages` itself; `base_url="https://argolink.io/v1"` produces `/v1/v1/messages`. `models.py` strips the `/v1`.
2. **`ChatOpenAI` switches endpoints on its own.** Passing `reasoning={...}` sends the request to `/v1/responses` even without `use_responses_api=True`; passing `reasoning_effort=` keeps it on `/v1/chat/completions`. If a model only behaves on one of them, set `use_responses_api` explicitly.
3. **Leave sampling parameters unset on Claude.** `claude-sonnet-5` returns 400 for any non-default `temperature`, `top_p` or `top_k`. `ChatAnthropic` sends none of them unless you set them.

## Workflow notes

**1. Streaming chat.** `chat_model("glm-5.3-flash", effort="high", stream_usage=True)` streams Chat Completions deltas and reports usage on the last chunk. GLM-5.3 Flash always thinks; `low` runs as `high`, so `high` and `max` are its real levels.

**2. Structured extraction.** The Pydantic model becomes a strict JSON Schema in the Responses API's `text.format`. `include_raw=True` returns `{"raw", "parsed", "parsing_error"}` instead of raising, so `extract()` retries an unparseable reply and logs it.

```python
llm = chat_model("gpt-6-sol", effort="low")
runnable = llm.with_structured_output(ActionItems, method="json_schema", include_raw=True)
```

**3. Tool-calling agent.** `create_agent(model, tools=[...], system_prompt=...)` from `langchain.agents` runs the loop on LangGraph: model → tool calls → `function_call_output` → model, until the model answers in text. The test drives it through a mocked `/v1/responses` and checks the tool result is sent back.

**4. Triage graph.** A cheap structured classifier decides the branch; only bugs and urgent tickets reach the stronger, pricier model. Every model is injectable, so the routing is tested without a network.

**5. Map-reduce.** `deepseek-v4.1-flash` (1,048,576-token context, thinking turned off with `reasoning_effort: "none"`) summarizes ~8K-token chunks in parallel; `gpt-6-sol` merges them. Mind the clock: DeepSeek models bill 2× on weekdays 09:00–12:00 and 14:00–18:00 Beijing time.

## Structured output: what each class sends

| Class and method | Request field | Notes |
|---|---|---|
| `ChatOpenAI` on Responses, `method="json_schema"` | `text.format = {"type": "json_schema", "strict": true, ...}` | used by workflow 2 |
| `ChatOpenAI` on Chat Completions, `method="json_schema"` | `response_format = {"type": "json_schema", ...}` | for `deepseek-*`, `glm-*`, `kimi-*` |
| `ChatAnthropic`, `method="function_calling"` (default) | a tool plus `tool_choice: {"type": "tool"}` | `claude-opus-5-5` and `claude-fable-5-1` return an error when a tool is forced |
| `ChatAnthropic`, `method="json_schema"` | `output_config.format` | not covered by the endpoint's model pages; test it on your key first |

Sending `response_format` to a Claude model over Chat Completions or Responses does not help: the model pages say it is not carried over to Anthropic Messages. Whatever the method, keep validation on your side; `include_raw=True` makes that cheap.

## Models and prices

USD per 1M tokens. Reasoning tokens bill as output.

| Model | Used in | Input | Cached input | Output | Official input / output | Effort values |
|---|---|---:|---:|---:|---:|---|
| [`gpt-6-sol`](https://argolink.io/gh-llm-application-workflows) | 2, 3, 5 | $0.60 | $0.06 | $3.00 | $2 / $10 | low · medium · high · xhigh · max |
| `gpt-5.6-luna` | 4 | $0.06 | $0.006 | $0.36 | $0.20 / $1.20 | low · medium · high · xhigh · max |
| `gpt-5.6-terra` | 4 | $0.60 | $0.06 | $3.60 | $2 / $12 | low · medium · high · xhigh · max |
| `claude-sonnet-5` | 4 | $0.80 | $0.08 | $4.00 | $2 / $10 | low · medium · high · xhigh · max |
| `deepseek-v4.1-flash` | 5 | $0.12 | $0.0024 | $0.48 | $0.15 / $0.60 | none · low · high · max |
| `glm-5.3-flash` | 1 | $0.12 | $0.024 | $0.40 | $0.15 / $0.50 | high · max |

<sub>Effective rates and official list prices from the public catalog (`GET https://argolink.io/api/catalog/v1/models?page_size=100`) and the model pages, 2026-10-07. GPT models bill a prompt over 272K tokens at 2× input and 1.5× output. Re-check before you budget.</sub>

## What each workflow costs

Token counts are assumptions stated in the row; reasoning tokens come on top of the output shown.

| Workflow | Assumption | Cost |
|---|---|---:|
| 1. Streaming chat | 50 in, 600 out on `glm-5.3-flash` | $0.00025 |
| 2. Extraction | 1,000 notes × (2,000 in, 300 out) on `gpt-6-sol` | $2.10 |
| 3. Support agent | 2 calls × (700 in, 70 out) on `gpt-6-sol` | $0.0013 |
| 4. Triage, classify step | 300 in, 20 out on `gpt-5.6-luna` | $0.000025 |
| 4. Triage, reply / escalate | 400 in + 250 out on `gpt-5.6-terra` / 400 in + 400 out on `claude-sonnet-5` | $0.0011 / $0.0019 |
| 5. Map-reduce, 200K-token document | 25 chunks × (8,000 in, 400 out) on `deepseek-v4.1-flash` + reduce (10,000 in, 1,500 out) on `gpt-6-sol` | $0.039 off-peak, $0.068 peak |
| 5. Same document in one call | 200,000 in, 1,500 out on `gpt-6-sol` | $0.125 |

## FAQ

**Can I use `ChatAnthropic` with non-Claude models?**
Yes for Grok, DeepSeek, GLM and Kimi: the endpoint converts Anthropic Messages to their native format. Not for OpenAI models; `/v1/messages` is not open for them.

**Why does GPT-6 Sol ignore my tools on Chat Completions?**
On Chat Completions it calls functions only with `reasoning_effort: "none"`. For tools with reasoning use the Responses API, which `chat_model()` picks for every `gpt-*` model.

**Is there an embeddings model for RAG?**
No embedding models were in the catalog on 2026-10-07. Pair these chat models with a local or separately hosted embedding model.

**How do I stop LangChain from retrying twice before my own fallback?**
Pass `max_retries=0` through `chat_model(...)`; both `ChatOpenAI` and `ChatAnthropic` accept it.

**Do I need LangGraph for workflow 3?**
`create_agent` is built on LangGraph and installs with `langchain`; you only import `langgraph` yourself for custom graphs like workflow 4.

## Run the tests

```bash
pip install -r requirements.txt pytest
pytest tests/
```

The tests swap the HTTP layer for `httpx` mock transports, so they need no key and send nothing. They check the URL, headers and body of every family, the agent loop, the graph's routing and the chunker.

## License

MIT
