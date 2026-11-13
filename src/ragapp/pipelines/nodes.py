"""Reusable graph nodes shared across pipeline variants (shared infrastructure)."""

from ragapp.config import get_settings
from ragapp.schemas import AnswerPayload, Condition, GroundedAnswer, RetrievedChunk
from ragapp.shared.llm import get_generator_llm
from ragapp.shared.prompts import GROUNDED_SYSTEM, format_context
from ragapp.shared.vectorstore import dense_search


def make_dense_retrieve_node(k: int | None = None):
    def retrieve(state: dict) -> dict:
        s = get_settings()
        query = state.get("query") or state["question"]
        docs = dense_search(query, k=k or s.top_k)
        return {"documents": docs, "query": query}

    return retrieve


def build_grounded_payload(
    condition: Condition,
    state: dict,
    result: GroundedAnswer,
    extra_trace: dict | None = None,
) -> AnswerPayload:
    docs: list[RetrievedChunk] = state.get("documents", [])
    valid_ids = {d.chunk_id for d in docs}
    sentences = result.sentences if not result.refused else []
    # Keep only citations that point at actually-provided chunks; fabricated
    # chunk ids are recorded in the trace (they count against citation precision).
    fabricated = []
    for sent in sentences:
        bad = [c for c in sent.citations if c not in valid_ids]
        fabricated.extend(bad)
    cited = sorted({c for s_ in sentences for c in s_.citations})

    trace = dict(state.get("trace", {}))
    if extra_trace:
        trace.update(extra_trace)
    if fabricated:
        trace["fabricated_citations"] = fabricated

    return AnswerPayload(
        condition=condition,
        question_id=state.get("question_id", -1),
        answer_text=" ".join(s_.text for s_ in sentences).strip(),
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        sentences=sentences,
        cited_chunk_ids=cited,
        retrieved_chunks=docs,
        trace=trace,
    )


def make_grounded_generate_node(condition: Condition, system_prompt: str = GROUNDED_SYSTEM):
    """Standard grounded generation with sentence-level citations (C1/C2)."""

    def generate(state: dict) -> dict:
        llm = get_generator_llm().with_structured_output(GroundedAnswer)
        context = format_context(state.get("documents", []))
        result: GroundedAnswer = llm.invoke(
            [
                ("system", system_prompt),
                ("human", f"Context passages:\n\n{context}\n\nQuestion: {state['question']}"),
            ]
        )
        return {"payload": build_grounded_payload(condition, state, result)}

    return generate
