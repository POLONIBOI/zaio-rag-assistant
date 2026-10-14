"""Retrieval check for the live (Netlify) version, on the same labelled questions.

    python scripts/eval_live_index.py      # needs netlify/data/index.json and OPENAI_API_KEY

The live version searches OpenAI embeddings instead of MiniLM ones, so the
"not found" cut-off has to be set again. This is what MIN_SIMILARITY in
netlify/lib/rag.mjs is based on. Embedding the 64 questions costs well under
a cent.
"""

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_retrieval import HANDBOOK_QUESTIONS, UNANSWERABLE, WEBSITE_QUESTIONS, url_matches  # noqa: E402

PER_SOURCE = 5


def embed(texts):
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": "text-embedding-3-small", "input": texts},
        timeout=60,
    )
    response.raise_for_status()
    return [d["embedding"] for d in sorted(response.json()["data"], key=lambda d: d["index"])]


def search(vector, index):
    """Same as search() in netlify/lib/rag.mjs: best 5 per source, merged."""
    hits = []
    for source in ("handbook", "website"):
        scored = [(sum(a * b for a, b in zip(vector, c["vector"])), c) for c in index if c["source"] == source]
        hits += sorted(scored, key=lambda h: h[0], reverse=True)[:PER_SOURCE]
    return sorted(hits, key=lambda h: h[0], reverse=True)


def main():
    load_dotenv(ROOT / ".env")
    index = json.loads((ROOT / "netlify" / "data" / "index.json").read_text(encoding="utf-8"))
    questions = [q for q, _ in HANDBOOK_QUESTIONS] + [q for q, _ in WEBSITE_QUESTIONS] + UNANSWERABLE
    vectors = embed(questions)

    best_answerable, misses = [], []
    for i, (question, page) in enumerate(HANDBOOK_QUESTIONS):
        hits = search(vectors[i], index)
        best_answerable.append((hits[0][0], question))
        if page not in [c["page"] for _, c in hits if c["source"] == "handbook"]:
            misses.append(question)
    offset = len(HANDBOOK_QUESTIONS)
    for i, (question, accepted) in enumerate(WEBSITE_QUESTIONS):
        hits = search(vectors[offset + i], index)
        best_answerable.append((hits[0][0], question))
        if not any(url_matches(c["url"], accepted) for _, c in hits if c["source"] == "website"):
            misses.append(question)

    offset += len(WEBSITE_QUESTIONS)
    print("Unanswerable: best similarity")
    best_unanswerable = []
    for i, question in enumerate(UNANSWERABLE):
        score = search(vectors[offset + i], index)[0][0]
        best_unanswerable.append(score)
        print(f"  {score:.2f}  {question}")

    print("\nLowest answerable questions: best similarity")
    for score, question in sorted(best_answerable)[:5]:
        print(f"  {score:.2f}  {question}")

    n = len(HANDBOOK_QUESTIONS) + len(WEBSITE_QUESTIONS)
    print(f"\nRight page sent to the model: {n - len(misses)}/{n}" + (f"  (missed: {'; '.join(misses)})" if misses else ""))
    print(f"Lowest answerable: {min(s for s, _ in best_answerable):.2f}   "
          f"highest unanswerable: {max(best_unanswerable):.2f}")


if __name__ == "__main__":
    main()
