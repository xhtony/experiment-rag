"""Shared LangGraph state. All four condition graphs read/write this state, so
the experiment runner can treat them uniformly."""

from typing import TypedDict

from ragapp.schemas import AnswerPayload, RetrievedChunk


class RAGState(TypedDict, total=False):
    question: str
    question_id: int

    # retrieval
    query: str                     # current (possibly rewritten) search query
    documents: list[RetrievedChunk]

    # C3 corrective loop
    evidence_sufficient: bool
    missing_info: str
    retry_count: int

    # output
    payload: AnswerPayload

    # debug trace accumulated by nodes
    trace: dict
