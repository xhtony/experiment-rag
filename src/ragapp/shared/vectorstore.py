"""Qdrant vector store shared by all RAG conditions (C1-C3)."""

from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ragapp.config import get_settings
from ragapp.schemas import RetrievedChunk
from ragapp.shared.llm import get_embeddings

EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@lru_cache
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(recreate: bool = False) -> None:
    s = get_settings()
    client = get_qdrant_client()
    dim = EMBEDDING_DIMS.get(s.embedding_model, 1536)
    if recreate and client.collection_exists(s.qdrant_collection):
        client.delete_collection(s.qdrant_collection)
    if not client.collection_exists(s.qdrant_collection):
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=dim, distance=qmodels.Distance.COSINE
            ),
        )


@lru_cache
def get_vectorstore() -> QdrantVectorStore:
    s = get_settings()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=s.qdrant_collection,
        embedding=get_embeddings(),
    )


def dense_search(query: str, k: int) -> list[RetrievedChunk]:
    results = get_vectorstore().similarity_search_with_score(query, k=k)
    out = []
    for doc, score in results:
        meta = doc.metadata
        out.append(
            RetrievedChunk(
                chunk_id=meta.get("chunk_id", ""),
                document_title=meta.get("document_title", ""),
                section=meta.get("section", ""),
                text=doc.page_content,
                score=float(score),
                retriever="dense",
            )
        )
    return out
