"""C3 - Corrective / refusal-aware RAG.

Graph:
    retrieve -> grade_evidence --(sufficient)--> generate -> verify_answer -> END
                     |  (insufficient, retries left)             | (unsupported)
                     v                                           v
               rewrite_query -> retrieve (loop)                refuse -> END
                     |  (insufficient, no retries left)
                     v
                  refuse -> END

Experimental treatments relative to C1:
1. Evidence sufficiency check before generation (corrective retrieval with
   query rewriting when evidence is insufficient).
2. Explicit refusal rules in the generation prompt.
3. Post-generation verification: unsupported draft answers are converted into
   refusals rather than returned.
"""

from typing import Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ragapp.config import get_settings
from ragapp.schemas import AnswerPayload, Condition, GroundedAnswer
from ragapp.shared.llm import get_generator_llm
from ragapp.shared.prompts import (
    ANSWER_VERIFIER_SYSTEM,
    EVIDENCE_GRADER_SYSTEM,
    QUERY_REWRITER_SYSTEM,
    REFUSAL_AWARE_SYSTEM,
    format_context,
)
from ragapp.shared.vectorstore import dense_search
from ragapp.pipelines.nodes import build_grounded_payload
from ragapp.pipelines.state import RAGState


class _EvidenceGrade(BaseModel):
    sufficient: bool = Field(
        description="True if the passages contain enough information for a complete, correct answer."
    )
    missing_info: Optional[str] = Field(
        default=None, description="If insufficient, what information is missing."
    )


class _Verification(BaseModel):
    fully_supported: bool
    problem: Optional[str] = None


def _retrieve(state: RAGState) -> dict:
    s = get_settings()
    query = state.get("query") or state["question"]
    docs = dense_search(query, k=s.top_k)
    trace = dict(state.get("trace", {}))
    trace.setdefault("queries", []).append(query)
    return {"documents": docs, "query": query, "trace": trace}


def _grade_evidence(state: RAGState) -> dict:
    llm = get_generator_llm().with_structured_output(_EvidenceGrade)
    context = format_context(state["documents"])
    grade: _EvidenceGrade = llm.invoke(
        [
            ("system", EVIDENCE_GRADER_SYSTEM),
            ("human", f"Question: {state['question']}\n\nPassages:\n\n{context}"),
        ]
    )
    trace = dict(state.get("trace", {}))
    trace.setdefault("evidence_grades", []).append(
        {"sufficient": grade.sufficient, "missing_info": grade.missing_info}
    )
    return {
        "evidence_sufficient": grade.sufficient,
        "missing_info": grade.missing_info or "",
        "trace": trace,
    }


def _route_after_grade(state: RAGState) -> str:
    if state["evidence_sufficient"]:
        return "generate"
    if state.get("retry_count", 0) < get_settings().c3_max_retries:
        return "rewrite_query"
    return "refuse"


def _rewrite_query(state: RAGState) -> dict:
    llm = get_generator_llm()
    response = llm.invoke(
        [
            ("system", QUERY_REWRITER_SYSTEM),
            (
                "human",
                f"Original question: {state['question']}\n"
                f"Missing evidence: {state.get('missing_info', 'unknown')}",
            ),
        ]
    )
    return {
        "query": response.content.strip().strip('"'),
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _generate(state: RAGState) -> dict:
    llm = get_generator_llm().with_structured_output(GroundedAnswer)
    context = format_context(state["documents"])
    result: GroundedAnswer = llm.invoke(
        [
            ("system", REFUSAL_AWARE_SYSTEM),
            ("human", f"Context passages:\n\n{context}\n\nQuestion: {state['question']}"),
        ]
    )
    return {"payload": build_grounded_payload(Condition.C3, state, result)}


def _route_after_generate(state: RAGState) -> str:
    # Refused answers need no verification; everything else is checked.
    return "end" if state["payload"].refused else "verify"


def _verify_answer(state: RAGState) -> dict:
    payload: AnswerPayload = state["payload"]
    llm = get_generator_llm().with_structured_output(_Verification)
    context = format_context(payload.retrieved_chunks)
    verdict: _Verification = llm.invoke(
        [
            ("system", ANSWER_VERIFIER_SYSTEM),
            (
                "human",
                f"Context passages:\n\n{context}\n\n"
                f"Question: {state['question']}\n\nDraft answer: {payload.answer_text}",
            ),
        ]
    )
    trace = dict(payload.trace)
    trace["verification"] = verdict.model_dump()
    if not verdict.fully_supported:
        payload = payload.model_copy(
            update={
                "refused": True,
                "refusal_reason": (
                    "Answer withheld after verification: "
                    + (verdict.problem or "draft answer not fully supported by sources.")
                ),
                "answer_text": "",
                "sentences": [],
                "cited_chunk_ids": [],
                "trace": trace,
            }
        )
    else:
        payload = payload.model_copy(update={"trace": trace})
    return {"payload": payload}


def _refuse(state: RAGState) -> dict:
    payload = AnswerPayload(
        condition=Condition.C3,
        question_id=state.get("question_id", -1),
        answer_text="",
        refused=True,
        refusal_reason=(
            "Insufficient evidence in the indexed policy documents: "
            + (state.get("missing_info") or "no supporting passages found.")
        ),
        retrieved_chunks=state.get("documents", []),
        trace=state.get("trace", {}),
    )
    return {"payload": payload}


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", _retrieve)
    g.add_node("grade_evidence", _grade_evidence)
    g.add_node("rewrite_query", _rewrite_query)
    g.add_node("generate", _generate)
    g.add_node("verify_answer", _verify_answer)
    g.add_node("refuse", _refuse)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_evidence")
    g.add_conditional_edges(
        "grade_evidence",
        _route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query", "refuse": "refuse"},
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"verify": "verify_answer", "end": END},
    )
    g.add_edge("verify_answer", END)
    g.add_edge("refuse", END)
    return g.compile()
