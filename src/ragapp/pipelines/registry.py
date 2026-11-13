"""Condition registry: maps experimental conditions to compiled LangGraph graphs."""

from functools import lru_cache

from ragapp.schemas import Condition
from ragapp.pipelines import c0_no_rag, c1_vector_rag, c2_hybrid_rerank, c3_corrective_refusal

_BUILDERS = {
    Condition.C0: c0_no_rag.build_graph,
    Condition.C1: c1_vector_rag.build_graph,
    Condition.C2: c2_hybrid_rerank.build_graph,
    Condition.C3: c3_corrective_refusal.build_graph,
}

CONDITION_DESCRIPTIONS = {
    Condition.C0: "No-RAG LLM baseline",
    Condition.C1: "Standard vector RAG (dense top-k)",
    Condition.C2: "Hybrid retrieval (dense+BM25, RRF) + LLM reranking",
    Condition.C3: "Corrective / refusal-aware RAG",
}


@lru_cache
def get_graph(condition: Condition):
    return _BUILDERS[condition]()
