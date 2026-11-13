"""Pydantic schemas shared by pipelines and evaluation.

These define the *contract* between generation and evaluation: every pipeline,
regardless of strategy, must emit an `AnswerPayload`, which the unified
evaluator consumes.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    SINGLE_DOC = "single_doc"
    MULTI_DOC = "multi_doc"
    CONDITIONAL = "conditional"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"


class Condition(str, Enum):
    C0 = "C0"  # No-RAG LLM baseline
    C1 = "C1"  # Standard vector RAG
    C2 = "C2"  # Hybrid retrieval + reranking
    C3 = "C3"  # Corrective / refusal-aware RAG
    C4 = "C4"  # Phase 2 hybrid corrective / refusal-aware RAG


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_title: str
    section: str = ""
    text: str
    score: float = 0.0
    retriever: str = ""  # "dense" | "bm25" | "fused"


# ---------------------------------------------------------------------------
# Structured generation output
# ---------------------------------------------------------------------------

class CitedSentence(BaseModel):
    """One sentence of the answer with its supporting chunk ids.

    Sentence-level citations enable the citation-support metric
    (does each citation actually support the sentence it is attached to?).
    """

    text: str = Field(description="One sentence of the answer.")
    citations: List[str] = Field(
        default_factory=list,
        description="chunk_ids from the provided context that support this sentence.",
    )


class GroundedAnswer(BaseModel):
    """Structured output schema for RAG conditions (C1, C2, C3)."""

    refused: bool = Field(
        description="True if you cannot answer from the provided context and refuse instead."
    )
    refusal_reason: Optional[str] = Field(
        default=None, description="If refused, briefly explain why."
    )
    sentences: List[CitedSentence] = Field(
        default_factory=list,
        description="The answer, sentence by sentence, each with citations. Empty if refused.",
    )


class PlainAnswer(BaseModel):
    """Structured output schema for the no-RAG baseline (C0)."""

    refused: bool = Field(description="True if you decline to answer.")
    refusal_reason: Optional[str] = None
    answer: str = Field(default="", description="Your answer. Empty if refused.")


class AnswerPayload(BaseModel):
    """Unified record produced by every pipeline for storage + evaluation."""

    condition: Condition
    question_id: int
    answer_text: str
    refused: bool
    refusal_reason: Optional[str] = None
    sentences: List[CitedSentence] = Field(default_factory=list)
    cited_chunk_ids: List[str] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    latency_ms: int = 0
    trace: dict = Field(default_factory=dict)  # pipeline-specific debug info
