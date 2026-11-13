"""Independent human rubric packets with private identity maps and paired sampling."""

import csv
import json
import math
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from ragapp.phase2.models import Dataset, ManualBatch, ManualScore, Result, Run
from ragapp.storage.db import session_scope

QUALITY = ("reference_groundedness", "retrieved_context_faithfulness", "citation_precision",
           "citation_completeness", "hallucination_rate", "required_fact_coverage")
BINARY = ("hallucinated", "action_appropriate", "appropriate_clarification")
RUBRIC = (*QUALITY, *BINARY, "faithfulness_rubric", "error_category", "notes")
ERRORS = {"", "none", "retrieval_failure", "ranking_failure", "generation_failure", "citation_failure",
          "over_answering", "over_refusal", "inappropriate_clarification", "policy_interpretation_error",
          "verification_false_rejection", "undetermined"}


def _safe(value):
    value = str(value)
    return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value


def _evidence(chunks):
    return "\n\n".join(f"[{c['chunk_id']}] {c['document_title']} / {c.get('section', '')}\n{c['text']}" for c in chunks)


def select_questions(questions, sample_questions, seed):
    if sample_questions is None:
        return [q["id"] for q in questions]
    if sample_questions < 1 or sample_questions > len(questions):
        raise ValueError("sample_questions must be between 1 and dataset size")
    groups = defaultdict(list)
    for question in questions:
        groups[question["qtype"]].append(question["id"])
    if sample_questions < len(groups):
        raise ValueError("Stratified sampling requires at least one question per represented type")
    rng = random.Random(seed)
    for values in groups.values():
        values.sort()
        rng.shuffle(values)
    selected = []
    while len(selected) < sample_questions:
        types = sorted(groups)
        rng.shuffle(types)
        for qtype in types:
            if groups[qtype] and len(selected) < sample_questions:
                selected.append(groups[qtype].pop())
    return selected


def export_manual(run_id: int, out: Path, sample_questions: int | None = None, seed: int = 42) -> dict:
    out = Path(out)
    if out.exists():
        raise ValueError("Manual packet already exists; refusing overwrite")
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError("Unknown Phase 2 run")
        dataset = session.get(Dataset, run.dataset_id)
        if any(not q.get("gold_reviewed") for q in dataset.questions):
            raise ValueError("Manual evaluation requires human-reviewed gold annotations")
        questions = {q["id"]: q for q in dataset.questions}
        answers = session.query(Result).filter_by(run_id=run_id, status="success").all()
        by_cell = {(a.question_id, a.condition): a for a in answers}
        if any((cell["question_id"], cell["condition"]) not in by_cell for cell in run.schedule):
            raise ValueError("Manual export requires a complete paired run")
        chosen = set(select_questions(dataset.questions, sample_questions, seed))
        reference = {c["chunk_id"]: c for c in run.reference_chunks}
        batch_id, private, packet = str(uuid.uuid4()), {}, []
        for cell in run.schedule:
            if cell["question_id"] not in chosen:
                continue
            answer = by_cell[(cell["question_id"], cell["condition"])]
            payload, q = answer.payload, questions[answer.question_id]
            review_id = str(uuid.uuid4())
            fields = {"batch_id": batch_id, "review_id": review_id, "question": q["question"],
                      "qtype": q["qtype"], "gold_answer": q["gold_answer"],
                      "required_facts": json.dumps(q["required_facts"], ensure_ascii=False),
                      "required_fact_coverage_applicable": bool(q["required_facts"] and "ANSWER" in q["acceptable_actions"]),
                      "expected_action": q["expected_action"],
                      "acceptable_actions": json.dumps(q["acceptable_actions"]),
                      "missing_slots": json.dumps(q["missing_slots"], ensure_ascii=False),
                      "acceptable_clarifications": json.dumps(q["acceptable_clarifications"], ensure_ascii=False),
                      "annotation_notes": q["annotation_notes"],
                      "reference_evidence": _evidence([reference[cid] for cid in q["gold_chunk_ids"]]),
                      "background_reference_evidence": _evidence([reference[cid] for cid in q.get("background_chunk_ids", [])]),
                      "retrieved_context": _evidence(payload["retrieved_chunks"]),
                      "answer_text": payload["answer_text"], "action": payload["action"],
                      "reason": payload["reason"], "clarification_question": payload["clarification_question"],
                      "cited_sentences": json.dumps(payload["sentences"], ensure_ascii=False)}
            fields = {k: _safe(v) for k, v in fields.items()}
            private[review_id] = {"answer_id": answer.id, "static": fields}
            packet.append({**fields, **dict.fromkeys(RUBRIC, "")})
        random.Random(seed).shuffle(packet)
        session.add(ManualBatch(id=batch_id, run_id=run_id, seed=seed, records=private))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(packet[0]))
        writer.writeheader()
        writer.writerows(packet)
    return {"batch_id": batch_id, "rows": len(packet), "questions": len(chosen),
            "qtype_counts": dict(Counter(questions[qid]["qtype"] for qid in chosen)), "out": str(out)}


def _number(value, name):
    text = str(value or "").strip()
    if text.lower() in {"", "na", "n/a"}:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric rubric: {name}") from exc
    limit = 2 if name == "faithfulness_rubric" else 1
    if not math.isfinite(number) or number < 0 or number > limit:
        raise ValueError(f"Rubric {name} must be within 0..{limit}")
    if name in BINARY or name == "faithfulness_rubric":
        if not number.is_integer():
            raise ValueError(f"Rubric {name} requires integer categories")
        return int(number)
    return number


def import_manual(file: Path, evaluator: str, source: str = "human", attest_human: bool = False) -> int:
    if not evaluator.strip() or len(evaluator) > 128:
        raise ValueError("A nonblank evaluator identifier of at most 128 characters is required")
    if source not in {"human", "ai_assisted"}:
        raise ValueError("Unknown manual-score source")
    if source == "human" and not attest_human:
        raise ValueError("Independent human ratings require --attest-human")
    with Path(file).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing or duplicate CSV column names")
        records = list(reader)
    if not records or len({r.get("batch_id") for r in records}) != 1:
        raise ValueError("Import exactly one nonempty manual batch")
    if len({r.get("review_id") for r in records}) != len(records):
        raise ValueError("Duplicate review IDs")
    count = 0
    with session_scope() as session:
        batch = session.get(ManualBatch, records[0]["batch_id"])
        if batch is None:
            raise ValueError("Unknown manual batch")
        for record in records:
            mapping = batch.records.get(record.get("review_id"))
            if mapping is None:
                raise ValueError("Unknown review ID in this batch")
            expected = mapping["static"]
            if set(record) != set(expected) | set(RUBRIC):
                raise ValueError("Manual packet columns were changed")
            if any(str(record[k]).replace("\r\n", "\n") != v.replace("\r\n", "\n") for k, v in expected.items()):
                raise ValueError("Non-rubric packet fields were changed")
            if not any(str(record.get(k) or "").strip() for k in RUBRIC):
                continue
            scores = {name: _number(record.get(name), name) for name in (*QUALITY, *BINARY, "faithfulness_rubric")}
            if scores["required_fact_coverage"] is not None and expected.get("required_fact_coverage_applicable") != "True":
                raise ValueError("required_fact_coverage is N/A when there are no answerable required facts")
            scores["error_category"] = str(record.get("error_category") or "").strip()
            scores["notes"] = str(record.get("notes") or "")
            if scores["error_category"] not in ERRORS:
                raise ValueError("Unknown error_category")
            answer = session.get(Result, mapping["answer_id"])
            if answer is None or answer.run_id != batch.run_id or answer.status != "success":
                raise ValueError("Packet answer no longer matches its completed run")
            if answer.condition == "C0" and any(scores[x] is not None for x in (
                "citation_precision", "citation_completeness", "retrieved_context_faithfulness"
            )):
                raise ValueError("C0 has no retrieved-context/citation metrics; leave them blank")
            if answer.payload["action"] != "ANSWER" and any(scores[x] is not None for x in (
                "citation_precision", "citation_completeness"
            )):
                raise ValueError("Non-ANSWER responses have no citation metrics; leave them blank")
            rate, flag = scores["hallucination_rate"], scores["hallucinated"]
            if rate is not None and flag is not None and (rate > 0) != bool(flag):
                raise ValueError("Hallucination rate and binary flag contradict one another")
            existing = session.query(ManualScore).filter_by(answer_id=answer.id, evaluator=evaluator).one_or_none()
            if existing:
                if existing.scores == scores and existing.source == source:
                    continue
                raise ValueError("Existing ratings are immutable; use an explicit new evaluator/revision identifier")
            session.add(ManualScore(answer_id=answer.id, evaluator=evaluator, source=source,
                                    scores=scores, batch_id=batch.id))
            count += 1
    return count
