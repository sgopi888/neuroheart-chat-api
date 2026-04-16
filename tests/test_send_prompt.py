"""
Send the exact prompt from inputtollm.txt to gpt-5-nano and print the response.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

# Read the input file
with open("app/inputtollm.txt") as f:
    raw = f.read()

# Parse into structured messages — split on "system" markers
# The file has: system prompt, then system HRV blocks, then RAG, then query
lines = raw.strip().split("\n")

# Build messages from the file content
messages = []
current_role = None
current_content = []

i = 0
# Skip "Input" header
if lines[0].strip() == "Input":
    i = 1

# First block: system prompt (line 1)
system_prompt = lines[i].strip()
messages.append({"role": "system", "content": system_prompt})
i += 1

# Skip empty lines
while i < len(lines) and not lines[i].strip():
    i += 1

# Collect all remaining content until "Query:" as system context
context_lines = []
query = ""
while i < len(lines):
    line = lines[i]
    if line.strip().startswith("Query:"):
        # Next non-empty line is the query
        query_text = line.replace("Query:", "").strip()
        if not query_text:
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines):
                query = lines[i].strip()
        else:
            query = query_text
        break
    context_lines.append(line)
    i += 1

# Add context as a single system message
context = "\n".join(context_lines).strip()
if context:
    messages.append({"role": "system", "content": context})

# Add user query
if query:
    messages.append({"role": "user", "content": query})

print(f"Model: {MODEL}")
print(f"Messages: {len(messages)}")
for m in messages:
    role = m["role"]
    content = m["content"][:100] + "..." if len(m["content"]) > 100 else m["content"]
    print(f"  [{role}] {content}")
print(f"\nUser query: {query}")
print()

t0 = time.time()
resp = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_completion_tokens=16384,
)
latency = time.time() - t0

choice = resp.choices[0]
content = (choice.message.content or "").strip()

print(f"Finish reason: {choice.finish_reason}")
print(f"Usage: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}, total={resp.usage.total_tokens}")
print(f"Latency: {latency:.1f}s")
print(f"\nReply ({len(content)} chars):")
print(content if content else "(EMPTY)")
