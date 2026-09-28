"""ChromaDB, persisted to disk in chroma_db/."""

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from app import config


class IndexNotBuiltError(RuntimeError):
    def __init__(self):
        super().__init__("The knowledge base hasn't been indexed yet. Run: python -m app.ingest")


def _client():
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def reset_collection():
    """Drop and recreate the collection so re-running ingest never leaves stale chunks."""
    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except NotFoundError:
        pass
    # embedding_function=None: we always pass our own vectors, so Chroma
    # shouldn't download its default model.
    return client.create_collection(
        config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def get_collection():
    try:
        collection = _client().get_collection(config.COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        raise IndexNotBuiltError() from None
    if collection.count() == 0:
        raise IndexNotBuiltError()
    return collection
