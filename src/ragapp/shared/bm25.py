"""Sparse BM25 retriever over the same chunk corpus (used by C2 hybrid retrieval).

The index is built from the chunks table in PostgreSQL so dense and sparse
retrieval are guaranteed to operate over identical units.
"""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from ragapp.schemas import RetrievedChunk
from ragapp.storage.db import session_scope
from ragapp.storage.models import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self._chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for i in ranked:
            c = self._chunks[i]
            out.append(
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    document_title=c["document_title"],
                    section=c["section"],
                    text=c["text"],
                    score=float(scores[i]),
                    retriever="bm25",
                )
            )
        return out


@lru_cache
def get_bm25_index() -> BM25Index:
    with session_scope() as session:
        rows = session.query(Chunk).all()
        chunks = [
            {
                "chunk_id": r.id,
                "document_title": r.document.title,
                "section": r.section,
                "text": r.text,
            }
            for r in rows
        ]
    if not chunks:
        raise RuntimeError("No chunks in database - run ingestion first.")
    return BM25Index(chunks)
