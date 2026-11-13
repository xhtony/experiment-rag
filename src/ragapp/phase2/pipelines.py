"""Five controlled LangGraph variants with shared retrieval and answer-control nodes."""

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from ragapp.config import get_settings
from ragapp.phase2 import prompts
from ragapp.phase2.contracts import Action, PipelineResult
from ragapp.phase2.llm import StructuredClient
from ragapp.pipelines.c2_hybrid_rerank import rrf_fuse
from ragapp.schemas import CitedSentence, Condition, RetrievedChunk
from ragapp.shared.bm25 import get_bm25_index
from ragapp.shared.prompts import format_context
from ragapp.shared.vectorstore import dense_search


class Decision(BaseModel):
    action: Action
    reason: str = ""
    clarification_question: str = ""
    sentences: list[CitedSentence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == Action.ANSWER:
            if not self.sentences or any(not s.text.strip() for s in self.sentences):
                raise ValueError("ANSWER requires nonempty factual sentences")
            if self.clarification_question.strip():
                raise ValueError("ANSWER cannot also request clarification")
        else:
            if self.sentences:
                raise ValueError("CLARIFY/REFUSE must not contain an answer")
            if self.action == Action.CLARIFY and not self.clarification_question.strip():
                raise ValueError("CLARIFY requires a targeted question")
            if self.action == Action.REFUSE and (not self.reason.strip() or self.clarification_question.strip()):
                raise ValueError("REFUSE needs a reason and no clarification question")
        return self


class EvidenceGrade(BaseModel):
    sufficient: bool
    needs_clarification: bool
    missing_info: str = ""
    clarification_question: str = ""

    @model_validator(mode="after")
    def valid_grade(self):
        if self.needs_clarification and not self.clarification_question.strip():
            raise ValueError("Clarification grade lacks a question")
        if self.needs_clarification and self.sufficient:
            raise ValueError("Evidence grade cannot both require clarification and be sufficient")
        return self


class QueryRewrite(BaseModel):
    query: str = Field(min_length=1)


class RankedIds(BaseModel):
    ranked_chunk_ids: list[str]


class Verification(BaseModel):
    fully_supported: bool
    problem: str = ""


class State(TypedDict, total=False):
    question: str
    question_id: int
    query: str
    documents: list[RetrievedChunk]
    retries: int
    grade: EvidenceGrade
    decision: Decision
    trace: dict


def bounded_context(documents, top_k, max_chars):
    kept, seen = [], set()
    for document in documents:
        if document.chunk_id in seen:
            continue
        seen.add(document.chunk_id)
        if len(kept) == top_k:
            break
        used = len(format_context(kept)) + (2 if kept else 0)
        header_size = len(format_context([document.model_copy(update={"text": ""})]))
        remaining = max_chars - used - header_size
        if remaining <= 0:
            break
        kept.append(document.model_copy(update={"text": document.text[:remaining]}))
    return kept


def build_graph(condition, client=None, dense=None, sparse=None, settings=None):
    condition = Condition(condition)
    settings = settings or get_settings()
    client = client or StructuredClient()
    dense = dense or dense_search
    sparse = sparse or (lambda query, k: get_bm25_index().search(query, k))
    hybrid = condition in {Condition.C2, Condition.C4}
    corrective = condition in {Condition.C3, Condition.C4}

    def call(schema, prompt, content, purpose):
        return schema.model_validate(client.invoke(schema, [("system", prompt), ("human", content)], purpose))

    def retrieve(state):
        query = state.get("query") or state["question"]
        dense_docs = dense(query, k=settings.candidate_k if hybrid else settings.top_k)
        trace = dict(state.get("trace", {}))
        record = {"query": query, "dense_ids": [c.chunk_id for c in dense_docs], "strategy": "hybrid" if hybrid else "dense"}
        if hybrid:
            sparse_docs = sparse(query, k=settings.candidate_k)
            fused = rrf_fuse([dense_docs, sparse_docs], settings.rrf_k)[:settings.candidate_k]
            record.update(bm25_ids=[c.chunk_id for c in sparse_docs], fused_ids=[c.chunk_id for c in fused])
            record["candidate_ids"] = [c.chunk_id for c in fused]
            if fused:
                ranked = call(RankedIds, prompts.RERANK,
                    f"Original question: {state['question']}\nCandidate passages:\n{format_context(fused)}", "rerank")
                by_id, selected = {c.chunk_id: c for c in fused}, []
                seen = set()
                for cid in ranked.ranked_chunk_ids:
                    if cid in by_id and cid not in seen:
                        selected.append(by_id[cid])
                        seen.add(cid)
                selected += [c for c in fused if c.chunk_id not in seen]
                record["reranker_unknown_ids"] = [cid for cid in ranked.ranked_chunk_ids if cid not in by_id]
                record["reranked_ids"] = [c.chunk_id for c in selected]
            else:
                selected = []
        else:
            selected = dense_docs
            record["candidate_ids"] = [c.chunk_id for c in selected]
        documents = bounded_context(selected, settings.top_k, settings.max_context_chars)
        record["final_ids"] = [c.chunk_id for c in documents]
        record["context_chars"] = len(format_context(documents))
        trace["retrieval_attempts"] = [*trace.get("retrieval_attempts", []), record]
        return {"documents": documents, "query": query, "trace": trace}

    def grade(state):
        decision = call(EvidenceGrade, prompts.GRADE,
            f"Question: {state['question']}\nRetrieved passages:\n{format_context(state['documents'])}", "grade_evidence")
        trace = dict(state["trace"])
        trace["evidence_grades"] = [*trace.get("evidence_grades", []), decision.model_dump()]
        return {"grade": decision, "trace": trace}

    def after_grade(state):
        grade = state["grade"]
        if grade.needs_clarification:
            return "clarify"
        if grade.sufficient:
            return "generate"
        return "rewrite" if state.get("retries", 0) < settings.c3_max_retries else "refuse"

    def rewrite(state):
        query = call(QueryRewrite, prompts.REWRITE,
            f"Original question: {state['question']}\nMissing evidence: {state['grade'].missing_info}", "rewrite_query")
        if not query.query.strip():
            raise ValueError("Empty rewritten query")
        trace = dict(state["trace"])
        trace["rewrites"] = [*trace.get("rewrites", []), query.query.strip()]
        return {"query": query.query.strip(), "retries": state.get("retries", 0) + 1, "trace": trace}

    def generate(state):
        prompt = prompts.C0 if condition == Condition.C0 else prompts.CORRECTIVE if corrective else prompts.STANDARD
        content = f"Question: {state['question']}"
        if condition != Condition.C0:
            content += f"\nContext passages:\n{format_context(state.get('documents', []))}"
        decision = call(Decision, prompt, content, "generate")
        return {"decision": decision}

    def verify(state):
        draft = state["decision"]
        verdict = call(Verification, prompts.VERIFY,
            f"Question: {state['question']}\nContext:\n{format_context(state['documents'])}\nDraft:\n{draft.model_dump_json()}", "verify_draft")
        trace = dict(state["trace"])
        trace.update(draft_before_verification=draft.model_dump(mode="json"), verification=verdict.model_dump())
        if not verdict.fully_supported:
            draft = Decision(action=Action.REFUSE, reason="The draft could not be fully supported by the available evidence.")
        return {"decision": draft, "trace": trace}

    def clarify(state):
        return {"decision": Decision(action=Action.CLARIFY, clarification_question=state["grade"].clarification_question)}

    def refuse(state):
        return {"decision": Decision(action=Action.REFUSE, reason="The retrieved documents do not provide sufficient evidence for a reliable answer.")}

    graph = StateGraph(State)
    graph.add_node("generate", generate)
    if condition == Condition.C0:
        graph.add_edge(START, "generate")
        graph.add_edge("generate", END)
    else:
        """retrieval part"""
        graph.add_node("retrieve", retrieve)
        graph.add_edge(START, "retrieve")
        if not corrective:
            graph.add_edge("retrieve", "generate")
            graph.add_edge("generate", END)
        else:
            """corrective part"""
            for name, node in (("grade_evidence", grade), ("rewrite", rewrite), ("verify", verify), ("clarify", clarify), ("refuse", refuse)):
                graph.add_node(name, node)
            graph.add_edge("retrieve", "grade_evidence")
            graph.add_conditional_edges("grade_evidence", after_grade, {x: x for x in ("generate", "rewrite", "clarify", "refuse")})
            graph.add_edge("rewrite", "retrieve")
            graph.add_conditional_edges("generate", lambda state: "verify" if state["decision"].action == Action.ANSWER else "end",
                                        {"verify": "verify", "end": END})
            for node in ("verify", "clarify", "refuse"):
                graph.add_edge(node, END)
    return graph.compile()


def run_pipeline(condition, question_id: int, question: str, *, client=None, dense=None, sparse=None, settings=None) -> PipelineResult:
    client = client or StructuredClient()
    settings = settings or get_settings()
    start = time.perf_counter()
    final = build_graph(condition, client, dense, sparse, settings).invoke(
        {"question": question, "question_id": question_id, "retries": 0, "trace": {}},
        config={"recursion_limit": 20 + settings.c3_max_retries * 5})
    decision = final["decision"]
    documents = final.get("documents", [])
    trace = final.get("trace", {})
    valid = {c.chunk_id for c in documents}
    trace["invalid_citation_ids"] = sorted({cid for s in decision.sentences for cid in s.citations if cid not in valid})
    return PipelineResult(condition=condition, question_id=question_id, action=decision.action,
        answer_text=" ".join(s.text for s in decision.sentences), reason=decision.reason,
        clarification_question=decision.clarification_question, sentences=decision.sentences,
        retrieved_chunks=documents, trace=trace, usage=client.summary(), latency_ms=int((time.perf_counter()-start)*1000))
