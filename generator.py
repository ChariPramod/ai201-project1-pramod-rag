"""
Grounded generation via Groq's Llama 3.3.

Pipeline:
  1. Retrieve top-k chunks via retriever.retrieve()
  2. Drop any chunk whose distance > DISTANCE_THRESHOLD
  3. If none survive, refuse WITHOUT calling the LLM
  4. Otherwise, build a source-labeled context block, send system+user messages,
     and return the answer alongside a deduped source list pulled from chunk
     metadata (not from the model's text).
"""

from groq import Groq

import retriever
from config import DISTANCE_THRESHOLD, GROQ_API_KEY, LLM_MODEL

SYSTEM_PROMPT = (
    "You are a study guide assistant. Answer the question using ONLY the context "
    "provided below. If the context does not contain enough information to answer, "
    "respond exactly: 'I don't have enough information on that.' Do not use any "
    "outside knowledge. If reviewers disagree, summarize the range of opinions "
    "rather than picking one."
)

REFUSAL = "I don't have enough information on that."

_client = Groq(api_key=GROQ_API_KEY)


def _build_context(chunks: list[dict]) -> str:
    blocks = [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
    return "\n\n".join(blocks)


def _dedupe_sources(chunks: list[dict]) -> list[str]:
    seen: list[str] = []
    for c in chunks:
        if c["source"] not in seen:
            seen.append(c["source"])
    return seen


def generate_response(query: str) -> dict:
    """Return {"answer": str, "sources": list[str]}. Refuses without an LLM
    call when no retrieved chunk meets DISTANCE_THRESHOLD."""
    chunks = retriever.retrieve(query)
    kept = [c for c in chunks if c["distance"] <= DISTANCE_THRESHOLD]

    if not kept:
        return {"answer": REFUSAL, "sources": []}

    user_message = (
        f"Context:\n{_build_context(kept)}\n\nQuestion: {query}"
    )

    completion = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = (completion.choices[0].message.content or "").strip()

    return {"answer": answer, "sources": _dedupe_sources(kept)}


if __name__ == "__main__":
    smoke_tests = [
        "Why do students say CS 70 feels artificially hard — harder than its content warrants?",
        "What is the average starting salary for CS 161 graduates?",
    ]
    for q in smoke_tests:
        print("=" * 80)
        print(f"Q: {q}")
        result = generate_response(q)
        print(f"\nA: {result['answer']}")
        print(f"\nSources: {result['sources']}")
        print()
