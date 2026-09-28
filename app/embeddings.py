"""One embedding model shared by indexing and querying.

Chunks and questions must be embedded by the same model or their vectors
aren't comparable, so both go through this module.
"""

from functools import lru_cache

from app import config


@lru_cache(maxsize=1)
def _model():
    # Imported here because torch takes a few seconds to load and most of the
    # unit tests never need it.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    # Normalised vectors, so cosine similarity is just the dot product and
    # scores are comparable across questions.
    return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_query(question: str) -> list[float]:
    return embed_texts([question])[0]
