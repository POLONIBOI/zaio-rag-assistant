"""How Retriever queries Chroma and merges the two sources, with a fake collection."""

from app.retriever import Retriever


class FakeCollection:
    """Returns canned rows per source and records every query."""

    def __init__(self, rows):
        self.rows = rows  # source -> [(text, metadata, distance)]
        self.queries = []

    def query(self, query_embeddings, n_results, where, include):
        self.queries.append({"n": n_results, "where": where})
        source = where["source"] if where else None
        rows = [r for s, rs in self.rows.items() if source in (None, s) for r in rs]
        rows = sorted(rows, key=lambda r: r[2])[:n_results]
        return {
            "documents": [[r[0] for r in rows]],
            "metadatas": [[r[1] for r in rows]],
            "distances": [[r[2] for r in rows]],
        }


ROWS = {
    "handbook": [
        ("Total Fees: R 38,950", {"source": "handbook", "section": "Fees", "page": 16}, 0.40),
        ("Loan declined? We try Manati.", {"source": "handbook", "section": "Fees", "page": 17}, 0.55),
    ],
    "website": [
        ("From R3,213/mo", {"source": "website", "section": "Tuition", "url": "https://www.zaio.io/compare-courses",
                            "title": "Compare Zaio Bootcamps"}, 0.30),
    ],
}


def make(**kwargs):
    collection = FakeCollection(ROWS)
    return Retriever(collection=collection, embed=lambda q: [0.0], **kwargs), collection


def test_each_source_is_queried_separately():
    retriever, collection = make(per_source=2, rerank=False)
    retriever.search("fees?")
    assert collection.queries == [{"n": 2, "where": {"source": "handbook"}},
                                  {"n": 2, "where": {"source": "website"}}]


def test_results_are_merged_best_similarity_first():
    retriever, _ = make(per_source=2, rerank=False)
    hits = retriever.search("fees?")

    assert [(h.source, h.similarity) for h in hits] == [("website", 0.7), ("handbook", 0.6), ("handbook", 0.45)]
    assert (hits[1].page, hits[1].url) == (16, None)
    assert (hits[0].page, hits[0].url, hits[0].title) == (None, "https://www.zaio.io/compare-courses",
                                                          "Compare Zaio Bootcamps")


def test_pooled_search_is_one_query():
    retriever, collection = make(per_source=None, top_k=2, rerank=False)
    hits = retriever.search("fees?")
    assert collection.queries == [{"n": 2, "where": None}]
    assert len(hits) == 2


def test_reranker_gets_the_merged_hits_and_decides_the_order():
    seen = []

    def reverse(question, hits):
        seen.append((question, [h.similarity for h in hits]))
        return list(reversed(hits))

    retriever, _ = make(per_source=2, rerank=reverse)
    hits = retriever.search("fees?")

    assert seen == [("fees?", [0.7, 0.6, 0.45])]
    assert [h.similarity for h in hits] == [0.45, 0.6, 0.7]
