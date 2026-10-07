"""Workflow 1: a streaming assistant turn with token accounting.

    python workflows/chat_stream.py "Explain the CAP theorem with one real example"
"""
import sys

from models import chat_model

llm = chat_model("glm-5.3-flash", effort="high", stream_usage=True)
prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain the CAP theorem with one real example."

usage = None
for chunk in llm.stream(prompt):
    print(chunk.text, end="", flush=True)
    usage = chunk.usage_metadata or usage
print()
if usage:
    print(f"\n{usage['input_tokens']} in / {usage['output_tokens']} out", file=sys.stderr)
