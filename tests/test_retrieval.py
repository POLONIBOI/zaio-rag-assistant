"""Retrieval against the real index: real embeddings, real ChromaDB, no LLM.

Skipped until `python -m app.ingest` has been run.
"""

import pytest

from app import config
from app.vector_store import IndexNotBuiltError, get_collection

try:
    get_collection()
except IndexNotBuiltError:
    pytest.skip("index not built - run python -m app.ingest", allow_module_level=True)

from app.retriever import Retriever  # noqa: E402


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


@pytest.mark.parametrize("question, page", [
    ("How much does the bootcamp cost?", 16),
    ("How is my final grade calculated?", 14),
    ("When are the live classes?", 11),
    ("What laptop do I need?", 5),
    ("What happens if my Capitec loan is declined?", 17),
    ("What are the tutor support hours?", 20),
    ("What do the colours on my calendar mean?", 9),
    ("Where do I submit my assignments?", 13),
])
def test_best_match_is_the_right_page(retriever, question, page):
    assert retriever.search(question)[0].page == page


def test_harder_question_is_within_top_k(retriever):
    # Page 24 only comes 5th here, which is why TOP_K is 5 and not 4.
    pages = [c.page for c in retriever.search("What mark do I need to get placed at a company?")]
    assert 24 in pages


def test_results_are_sorted_and_carry_metadata(retriever):
    hits = retriever.search("How much does the bootcamp cost?")
    assert len(hits) == config.TOP_K
    assert hits == sorted(hits, key=lambda h: h.similarity, reverse=True)
    assert hits[0].section == "Fees"


@pytest.mark.parametrize("question", ["What is the capital of France?", "How do I cook pasta?"])
def test_off_topic_questions_fall_under_the_threshold(retriever, question):
    assert retriever.search(question)[0].similarity < config.MIN_SIMILARITY
