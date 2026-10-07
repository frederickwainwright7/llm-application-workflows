"""Workflow 5: map-reduce summary of a long document.

Map: a cheap 1M-context model summarizes each chunk, several in parallel.
Reduce: a stronger model merges the partial summaries.

    python workflows/summarize_long.py annual_report.txt
"""
import sys

from models import chat_model

CHUNK_CHARS = 32_000  # about 8K tokens of English text


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries where possible."""
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        if current and len(current) + len(paragraph) > size:
            parts.append(current)
            current = ""
        current += paragraph + "\n\n"
        while len(current) > size:
            parts.append(current[:size])
            current = current[size:]
    if current.strip():
        parts.append(current)
    return parts


def summarize(text: str, mapper=None, reducer=None, concurrency: int = 4) -> str:
    mapper = mapper or chat_model("deepseek-v4.1-flash", effort="none")
    reducer = reducer or chat_model("gpt-6-sol", effort="low")
    prompts = [f"Summarize this part of a document in at most 8 bullet points. Keep numbers exact.\n\n{c}"
               for c in chunks(text)]
    partials = mapper.batch(prompts, config={"max_concurrency": concurrency})
    joined = "\n\n".join(f"Part {i + 1}:\n{p.text}" for i, p in enumerate(partials))
    final = reducer.invoke(f"Merge these partial summaries into one summary of at most 12 bullet points:\n\n{joined}")
    return final.text


if __name__ == "__main__":
    print(summarize(open(sys.argv[1]).read()))
