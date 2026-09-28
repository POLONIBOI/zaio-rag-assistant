"""Retrieval against the real index: real embeddings, real ChromaDB, real reranker, no LLM.

Skipped until `python -m app.ingest` has been run.
"""

import pytest

from app import config
from app.vector_store import IndexNotBuiltError, get_collection

try:
    get_collection()
except IndexNotBuiltError:
    pytest.skip("index not built - run python -m app.ingest", allow_module_level=True)

from app.retriever import HANDBOOK, WEBSITE, Retriever  # noqa: E402

SITE = "https://www.zaio.io"


@pytest.fixture(scope="module")
def retriever():
    return Retriever()


@pytest.fixture(scope="module")
def unranked():
    return Retriever(rerank=False)


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
def test_best_handbook_match_is_the_right_page(retriever, question, page):
    handbook_hits = [h for h in retriever.search(question) if h.source == HANDBOOK]
    assert handbook_hits[0].page == page


@pytest.mark.parametrize("question, path", [
    ("What is Zaio's refund policy?", "/terms"),
    ("How does Capitec education finance work?", "/capitec-finance"),
    ("Who is the CEO of Zaio?", "/aboutus"),
    ("What is The Weekly Commit?", "/newsletter"),
    ("When does the next Cloud & DevOps bootcamp start?", "/cloud-devops-engineer-bootcamp"),
    ("Where was the Zaio open day in Cape Town?", "/events/zaio-open-day-capetown"),
])
def test_best_website_match_is_the_right_page(retriever, question, path):
    website_hits = [h for h in retriever.search(question) if h.source == WEBSITE]
    assert website_hits[0].url == SITE + path


def test_both_sources_are_always_searched(retriever):
    # A handbook question still gets website excerpts and vice versa, so the
    # LLM can see when the two disagree (fees, start dates).
    for question in ["When is orientation day?", "What is Zaio's refund policy?"]:
        assert {h.source for h in retriever.search(question)} == {HANDBOOK, WEBSITE}


def test_page_24_reaches_the_llm_despite_website_text(retriever):
    # The reason for searching each source separately: in a pooled top 10 the
    # website's skills-development pages push page 24 out.
    pages = [h.page for h in retriever.search("What mark do I need to get placed at a company?")]
    assert 24 in pages


def test_reranking_moves_page_24_from_last_to_first(retriever, unranked):
    question = "What mark do I need to get placed at a company?"
    by_similarity = unranked.search(question)
    assert by_similarity[-1].page == 24  # 10th of 10, behind /skillsdevelopment
    assert retriever.search(question)[0].page == 24


def test_reranking_brings_the_laptop_email_into_the_local_models_top_3(retriever, unranked):
    question = "Who do I email about laptop advice?"  # asif@zaio.io is on page 5
    assert 5 not in [h.page for h in unranked.search(question)[:3]]
    assert 5 in [h.page for h in retriever.search(question)[:config.LOCAL_MAX_EXCERPTS]]


def test_results_carry_their_metadata(retriever):
    hits = retriever.search("What is Zaio's refund policy?")
    assert len(hits) == 2 * config.TOP_K_PER_SOURCE
    for hit in hits:
        if hit.source == HANDBOOK:
            assert hit.page and hit.url is None
            assert hit.label == f"Student Handbook - Page {hit.page}"
        else:
            assert hit.url.startswith(SITE) and hit.page is None
            assert hit.label == hit.url
        assert hit.rerank_score is not None
    assert [h.rerank_score for h in hits] == sorted((h.rerank_score for h in hits), reverse=True)


@pytest.mark.parametrize("question", ["What is the capital of France?", "How do I cook pasta?"])
def test_off_topic_questions_fall_under_the_threshold(retriever, question):
    assert max(h.similarity for h in retriever.search(question)) < config.MIN_SIMILARITY
