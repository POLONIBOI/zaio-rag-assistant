"""Question in, answer + source out. This is what POST /ask calls."""

from dataclasses import dataclass

from app import config
from app.retriever import HANDBOOK, RetrievedChunk


@dataclass(frozen=True)
class Answer:
    answer: str
    # "Student Handbook - Page 16", a zaio.io URL, or None when nothing was found
    source: str | None


NOT_FOUND = Answer(answer=config.NOT_FOUND_ANSWER, source=None)


def format_source(cited: list[RetrievedChunk]) -> str:
    """The single source shown with an answer.

    The first excerpt the model cited decides where the answer came from. For
    the handbook, every handbook page it used is listed ("Student Handbook -
    Pages 16, 17"), because fee answers often span two pages. For the website
    it's the URL of that page.
    """
    primary = cited[0]
    if primary.source != HANDBOOK:
        return primary.url
    pages = sorted({c.page for c in cited if c.source == HANDBOOK})
    if len(pages) == 1:
        return f"Student Handbook - Page {pages[0]}"
    return "Student Handbook - Pages " + ", ".join(str(p) for p in pages)


class Assistant:
    def __init__(self, retriever, llm, min_similarity: float = config.MIN_SIMILARITY):
        self._retriever = retriever
        self._llm = llm
        self._min_similarity = min_similarity

    def ask(self, question: str) -> Answer:
        chunks = self._retriever.search(question)

        # Nothing in either source is even close - don't spend an LLM call on it.
        # (max, not chunks[0]: after reranking the first chunk isn't always the
        # one with the highest similarity.)
        if not chunks or max(c.similarity for c in chunks) < self._min_similarity:
            return NOT_FOUND

        result = self._llm.answer(question, chunks)
        if not result.found or not result.answer.strip():
            # Always the exact same sentence, so n8n (or anything else) can
            # match on it instead of parsing whatever the model wrote.
            return NOT_FOUND

        # Only trust excerpt numbers we actually sent. If the model didn't
        # give a usable one, fall back to the best-matching chunk.
        cited = [chunks[i - 1] for i in dict.fromkeys(result.excerpts) if 1 <= i <= len(chunks)]
        return Answer(answer=result.answer.strip(), source=format_source(cited or [chunks[0]]))
