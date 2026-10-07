"""Workflow 2: turn meeting notes into validated action items (structured output).

GPT models get a strict JSON Schema through the Responses API (text.format).
include_raw=True keeps the raw message when parsing fails, so a bad reply is
logged and retried instead of crashing the batch.

    python workflows/extract.py notes.txt
"""
import json
import sys
from datetime import date

from pydantic import BaseModel, Field

from models import chat_model


class ActionItem(BaseModel):
    owner: str = Field(description="person responsible, as written in the notes")
    task: str = Field(description="what has to be done, imperative mood")
    due: date | None = Field(description="due date if one is stated, else null")


class ActionItems(BaseModel):
    items: list[ActionItem]


def extractor(model: str = "gpt-6-sol", **kwargs):
    llm = chat_model(model, effort="low", **kwargs)
    return llm.with_structured_output(ActionItems, method="json_schema", include_raw=True)


def extract(notes: str, runnable=None, attempts: int = 2) -> ActionItems:
    runnable = runnable or extractor()
    prompt = f"Extract every action item from these meeting notes.\n\n{notes}"
    for _ in range(attempts):
        result = runnable.invoke(prompt)
        if result["parsed"] is not None:
            return result["parsed"]
        print(f"unparseable reply, retrying: {result['parsing_error']}", file=sys.stderr)
    raise ValueError("no valid action items after retries")


if __name__ == "__main__":
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else (
        "Priya will send the revised SOW to Acme by Oct 14. Tom to fix the CSV export bug. "
        "We agreed Lena books the venue for the offsite before end of month.")
    print(json.dumps(extract(text).model_dump(mode="json"), indent=2))
