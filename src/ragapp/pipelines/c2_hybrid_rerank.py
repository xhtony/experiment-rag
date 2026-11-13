"""C2 - Hybrid retrieval (dense + BM25 with RRF fusion) + LLM reranking.

Graph:  hybrid_retrieve -> rerank -> generate -> END

- Dense (Qdrant) and sparse (BM25) each return `candidate_k` candidates.
- Candidates are fused with Reciprocal Rank Fusion.
- An LLM listwise reranker scores the fused pool; the top-k survive.
- Generation node is identical to C1 (controlled variable).
"""

from typing import List

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ragapp.config import get_settings
from ragapp.schemas import Condition, RetrievedChunk
from ragapp.shared.bm25 import get_bm25_index
from ragapp.shared.llm import get_generator_llm
from ragapp.shared.vectorstore import dense_search
from ragapp.pipelines.nodes import make_grounded_generate_node
from ragapp.pipelines.state import RAGState


def rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]], rrf_k: int
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievedChunk] = {}
    for ranking in ranked_lists:
        for rank, chunk in enumerate(ranking):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                rrf_k + rank + 1
            )
            best.setdefault(chunk.chunk_id, chunk)
    fused = []
    for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        c = best[cid].model_copy(update={"score": score, "retriever": "fused"})
        fused.append(c)
    return fused


def _hybrid_retrieve(state: RAGState) -> dict:
    s = get_settings()
    query = state.get("query") or state["question"]
    dense = dense_search(query, k=s.candidate_k)
    sparse = get_bm25_index().search(query, k=s.candidate_k)
    fused = rrf_fuse([dense, sparse], rrf_k=s.rrf_k)
    return {
        "documents": fused[: s.candidate_k],
        "query": query,
        "trace": {
            "dense_ids": [c.chunk_id for c in dense],
            "bm25_ids": [c.chunk_id for c in sparse],
            "fused_ids": [c.chunk_id for c in fused[: s.candidate_k]],
        },
    }


class _RerankResult(BaseModel):
    ranked_chunk_ids: List[str] = Field(
        description="chunk_ids ordered from most to least relevant to the question."
    )


_RERANK_SYSTEM = (
    "You are a relevance reranker for a university policy question-answering system. "
    "Given a question and candidate passages, return the chunk_ids ordered from most "
    "to least relevant. Judge relevance by whether the passage helps answer the "
    "question, including conditions and exceptions."
)


def _rerank(state: RAGState) -> dict:
    s = get_settings()
    docs = state["documents"]
    listing = "\n\n".join(
        f"[chunk_id={c.chunk_id}]\n{c.text[:600]}" for c in docs
    )
    llm = get_generator_llm().with_structured_output(_RerankResult)
    result: _RerankResult = llm.invoke(
        [
            ("system", _RERANK_SYSTEM),
            ("human", f"Question: {state['question']}\n\nCandidate passages:\n\n{listing}"),
        ]
    )
    by_id = {c.chunk_id: c for c in docs}
    reranked = [by_id[cid] for cid in result.ranked_chunk_ids if cid in by_id]
    # Append anything the LLM dropped, preserving fusion order, then cut to top_k.
    seen = {c.chunk_id for c in reranked}
    reranked += [c for c in docs if c.chunk_id not in seen]
    trace = dict(state.get("trace", {}))
    trace["reranked_ids"] = [c.chunk_id for c in reranked[: s.top_k]]
    return {"documents": reranked[: s.top_k], "trace": trace}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("hybrid_retrieve", _hybrid_retrieve)
    g.add_node("rerank", _rerank)
    g.add_node("generate", make_grounded_generate_node(Condition.C2))
    g.add_edge(START, "hybrid_retrieve")
    g.add_edge("hybrid_retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    return g.compile()
