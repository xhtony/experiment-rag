"""Unified automated evaluation.

Every stored answer (any condition) is scored on the four dependent variables:

1. Faithfulness      claim-level support against the *retrieved* context
                     (for C0, which retrieves nothing, claims are judged against
                     the gold evidence chunks -> measures factual grounding)
2. Citation accuracy precision/recall/F1 vs gold supporting chunks +
                     sentence-level citation support (LLM judge)
3. Hallucination     answer contains >= 1 unsupported/contradicted claim
4. Refusal behaviour confusion-matrix cell vs the gold answerability label

Plus retrieval diagnostics (hit / recall of gold chunks) for error analysis.
"""

from ragapp.evaluation.judges import (
    VERDICT_WEIGHT,
    citation_supports,
    decompose_claims,
    verify_claims,
)
from ragapp.storage.db import session_scope
from ragapp.storage.models import Answer, AutoScore, Chunk, Question


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or (p + r) == 0:
        return 0.0 if (p is not None and r is not None) else None
    return 2 * p * r / (p + r)


def _refusal_outcome(refused: bool, expect_refusal: bool) -> str:
    if expect_refusal and refused:
        return "correct_refusal"
    if expect_refusal and not refused:
        return "false_answer"      # should have refused but answered anyway
    if not expect_refusal and refused:
        return "false_refusal"     # answerable but refused
    return "correct_answer"


def _load_chunk_texts(session, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    rows = session.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
    return {r.id: r.text for r in rows}


def evaluate_answer(session, answer: Answer, question: Question) -> AutoScore:
    score = AutoScore(answer_id=answer.id)
    details: dict = {}

    # ----- reference context for faithfulness -----
    if answer.condition == "C0":
        ref_ids = list(question.gold_chunk_ids or [])
        details["reference"] = "gold_chunks"
    else:
        ref_ids = list(answer.retrieved_chunk_ids or [])
        details["reference"] = "retrieved_chunks"
    ref_texts = _load_chunk_texts(session, ref_ids)
    context = "\n\n".join(f"[chunk_id={cid}]\n{txt}" for cid, txt in ref_texts.items())

    # ----- refusal outcome (computed for every answer) -----
    score.refusal_outcome = _refusal_outcome(answer.refused, question.expect_refusal)

    # ----- retrieval diagnostics (RAG conditions only) -----
    gold_ids = set(question.gold_chunk_ids or [])
    if answer.condition != "C0":
        retrieved = set(answer.retrieved_chunk_ids or [])
        if gold_ids:
            hits = gold_ids & retrieved
            score.retrieval_hit = bool(hits)
            score.retrieval_recall = len(hits) / len(gold_ids)

    # Refused answers have no claims/citations to grade.
    if answer.refused or not answer.answer_text.strip():
        score.details = details
        return score

    # ----- faithfulness + hallucination -----
    claims = decompose_claims(answer.answer_text)
    verdicts = verify_claims(claims, context) if context else []
    score.total_claims = len(claims)
    if verdicts:
        weights = [VERDICT_WEIGHT[v.verdict] for v in verdicts]
        score.faithfulness = sum(weights) / len(weights)
        score.unsupported_claims = sum(
            1 for v in verdicts if v.verdict in ("unsupported", "partially_supported")
        )
        score.contradicted_claims = sum(1 for v in verdicts if v.verdict == "contradicted")
        score.hallucinated = any(
            v.verdict in ("unsupported", "contradicted") for v in verdicts
        )
        details["claim_verdicts"] = [v.model_dump() for v in verdicts]
    elif claims and not context:
        # Answered with no evidence at all: maximal hallucination case.
        score.faithfulness = 0.0
        score.unsupported_claims = len(claims)
        score.hallucinated = True
        details["claim_verdicts"] = []

    # ----- citation accuracy (RAG conditions only) -----
    if answer.condition != "C0":
        cited = set(answer.cited_chunk_ids or [])
        if gold_ids or cited:
            tp = len(cited & gold_ids)
            score.citation_precision = tp / len(cited) if cited else 0.0
            score.citation_recall = tp / len(gold_ids) if gold_ids else None
            score.citation_f1 = _f1(score.citation_precision, score.citation_recall)

        # Sentence-level: does each citation support its sentence?
        all_texts = _load_chunk_texts(
            session, sorted({c for s in (answer.sentences or []) for c in s.get("citations", [])})
        )
        checks, supported = 0, 0
        per_sentence = []
        for sent in answer.sentences or []:
            for cid in sent.get("citations", []):
                passage = all_texts.get(cid)
                if passage is None:  # fabricated citation id
                    ok = False
                else:
                    ok = citation_supports(sent["text"], passage)
                checks += 1
                supported += int(ok)
                per_sentence.append(
                    {"sentence": sent["text"], "chunk_id": cid, "supports": ok}
                )
        if checks:
            score.citation_support = supported / checks
            details["citation_checks"] = per_sentence

    score.details = details
    return score


def evaluate_run(run_id: int, conditions: list[str] | None = None) -> int:
    """Score all answers of a run that do not yet have an AutoScore."""
    n = 0
    with session_scope() as session:
        q = session.query(Answer).filter(Answer.run_id == run_id)
        if conditions:
            q = q.filter(Answer.condition.in_(conditions))
        answers = q.all()
        scored_ids = {
            s.answer_id
            for s in session.query(AutoScore.answer_id)
            .join(Answer, Answer.id == AutoScore.answer_id)
            .filter(Answer.run_id == run_id)
            .all()
        }
        questions = {qq.id: qq for qq in session.query(Question).all()}
        for ans in answers:
            if ans.id in scored_ids:
                continue
            score = evaluate_answer(session, ans, questions[ans.question_id])
            session.add(score)
            session.flush()
            n += 1
            print(
                f"  scored answer {ans.id} ({ans.condition}, Q{ans.question_id}): "
                f"faithfulness={score.faithfulness}, outcome={score.refusal_outcome}"
            )
    return n
