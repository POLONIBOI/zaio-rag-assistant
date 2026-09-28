"""Question in, answer + source out. This is what POST /ask calls."""

from dataclasses import dataclass

from app import config


@dataclass(frozen=True)
class Answer:
    answer: str
    source: str | None  # e.g. "Page 16", or None when nothing was found


NOT_FOUND = Answer(answer=config.NOT_FOUND_ANSWER, source=None)


def format_source(pages: list[int]) -> str:
    pages = sorted(set(pages))
    if len(pages) == 1:
        return f"Page {pages[0]}"
    return "Pages " + ", ".join(str(p) for p in pages)


class HandbookAssistant:
    def __init__(self, retriever, llm, min_similarity: float = config.MIN_SIMILARITY):
        self._retriever = retriever
        self._llm = llm
        self._min_similarity = min_similarity

    def ask(self, question: str) -> Answer:
        chunks = self._retriever.search(question)

        # Nothing in the handbook is even close - don't spend an API call on it.
        if not chunks or chunks[0].similarity < self._min_similarity:
            return NOT_FOUND

        result = self._llm.answer(question, chunks)
        if not result.found or not result.answer.strip():
            # Always the exact same sentence, so n8n (or anything else) can
            # match on it instead of parsing whatever the model wrote.
            return NOT_FOUND

        # Only trust page numbers we actually sent. If the model didn't give
        # any usable ones, fall back to the best-matching chunk.
        retrieved = {c.page for c in chunks}
        pages = [p for p in result.pages if p in retrieved] or [chunks[0].page]
        return Answer(answer=result.answer.strip(), source=format_source(pages))
