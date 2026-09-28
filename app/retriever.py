"""Part 2 (retrieval): embed the question and pull the closest chunks from ChromaDB."""

from dataclasses import dataclass

from app import config

HANDBOOK = "handbook"
WEBSITE = "website"
SOURCES = (HANDBOOK, WEBSITE)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str  # "handbook" or "website"
    section: str
    similarity: float  # cosine similarity, 1.0 = identical meaning
    page: int | None = None
    url: str | None = None
    title: str = ""
    rerank_score: float | None = None  # filled in by app/reranker.py

    @property
    def label(self) -> str:
        """How this chunk is cited in the API response."""
        if self.source == HANDBOOK:
            return f"Student Handbook - Page {self.page}"
        return self.url


class Retriever:
    """Searches both knowledge sources and returns the best chunks from each.

    With per_source set, each source is queried separately and the results
    are merged, so a question always gets the best handbook *and* the best
    website excerpts. A single pooled query lets whichever source has more
    text on a topic crowd the other out, and the website has about seven
    times as many chunks as the handbook.
    """

    def __init__(self, collection=None, embed=None, top_k: int = config.TOP_K,
                 per_source: int | None = config.TOP_K_PER_SOURCE, rerank=None):
        if collection is None:
            from app.vector_store import get_collection

            collection = get_collection()
        if embed is None:
            from app.embeddings import embed_query as embed
        if rerank is None:
            # rerank=False keeps plain similarity order (used by the eval script)
            from app.reranker import rerank as default_rerank

            rerank = default_rerank if config.RERANK_MODEL else False

        self._collection = collection
        self._embed = embed
        self._rerank = rerank
        self.top_k = top_k
        self.per_source = per_source

    def search(self, question: str) -> list[RetrievedChunk]:
        """Best matches for the question, best first."""
        vector = self._embed(question)
        if self.per_source:
            hits = [hit for source in SOURCES for hit in self._query(vector, self.per_source, source)]
        else:
            hits = self._query(vector, self.top_k)
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return self._rerank(question, hits) if self._rerank else hits

    def _query(self, vector, n: int, source: str | None = None) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=n,
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                text=doc,
                source=meta["source"],
                section=meta.get("section", ""),
                page=int(meta["page"]) if "page" in meta else None,
                url=meta.get("url"),
                title=meta.get("title", ""),
                # Collection uses cosine distance, which is 1 - similarity.
                similarity=round(1 - distance, 3),
            )
            for doc, meta, distance in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
