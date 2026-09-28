"""Second pass over the retrieved chunks: a cross-encoder puts the best ones first.

The embedding search compares a question vector with chunk vectors that were
made separately. A cross-encoder reads the question and a chunk together, so
it's slower (it runs once per chunk) but much better at telling which of the
10 candidates actually answers the question. On the 50 labelled questions in
scripts/eval_retrieval.py the right page comes first 44 times after
reranking, against 39 in similarity order, and is in the top 3 for 49.

That matters most for the local model, which only sees the top few excerpts.
"""

from dataclasses import replace
from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def rerank(question: str, chunks: list) -> list:
    """The same chunks, best first, each with its rerank score filled in."""
    if not chunks:
        return []
    pairs = [(question, f"{c.title or c.section}: {c.text}") for c in chunks]
    scores = _model().predict(pairs, show_progress_bar=False)
    scored = [replace(c, rerank_score=round(float(s), 2)) for c, s in zip(chunks, scores)]
    return sorted(scored, key=lambda c: c.rerank_score, reverse=True)
