"""Persistence adapters around independently testable evaluation and analysis."""

import copy
from pathlib import Path

import pandas as pd

from ragapp.config import get_settings
from ragapp.phase2.contracts import PipelineResult, QuestionRecord
from ragapp.phase2.evaluation_state import EVALUATION_PROTOCOL, METRIC_STAGE, evaluation_status
from ragapp.phase2.models import Dataset, EvaluationAttempt, ManualScore, Result, Run, Score
from ragapp.phase2.runner import execution_lock, run_status, verify_run
from ragapp.schemas import RetrievedChunk
from ragapp.storage.db import session_scope


def evaluate_run(run_id: int, allow_incomplete: bool = False, evaluator=None, retry_errors: bool = False):
    if evaluator is None:
        from ragapp.phase2.evaluation import evaluate_answer
        evaluator = evaluate_answer
    with execution_lock(run_id):
        run, dataset = verify_run(run_id, check_live_corpus=False)
        if run.manifest.get("evaluation_protocol") != EVALUATION_PROTOCOL:
            raise ValueError("Run was not frozen with the staged evaluation protocol; use a new versioned development run")
        status = run_status(run_id)
        if not allow_incomplete and status["successful"] != status["expected"]:
            raise ValueError("Run is incomplete; use --allow-incomplete only for explicitly labelled partial evaluation")
        questions = {q["id"]: QuestionRecord.model_validate(q) for q in dataset.questions}
        if any(not q.gold_reviewed for q in questions.values()):
            raise ValueError("Automatic evaluation requires human-reviewed gold labels, including development data")
        chunks = {c["chunk_id"]: RetrievedChunk.model_validate(c) for c in run.reference_chunks}
        budget = get_settings().phase2_max_attempts
        with session_scope() as session:
            answers = session.query(Result).filter_by(run_id=run_id, status="success").all()
            scores = {s.answer_id: s for s in session.query(Score).join(Result, Score.answer_id == Result.id).filter(Result.run_id == run_id)}
        for answer in answers:
            saved = scores.get(answer.id)
            if saved and evaluation_status(saved.metrics, saved.details) == "complete":
                continue
            previous = {"metrics": saved.metrics, "details": saved.details} if saved else None

            def persist(result):
                metrics, details = result["metrics"], result["details"]
                if evaluation_status(metrics, details) in {"invalid", "legacy_unversioned"}:
                    raise ValueError("Evaluator returned an invalid staged checkpoint")
                state = details["evaluation"]
                # Score snapshot and attempt journal commit together. Storage failures must propagate.
                with session_scope() as session:
                    row = session.query(Score).filter_by(answer_id=answer.id).one_or_none()
                    if row is None:
                        row = Score(answer_id=answer.id)
                    row.metrics, row.details = copy.deepcopy(metrics), copy.deepcopy(details)
                    row.provenance = {"judge_model": get_settings().judge_model, "protocol": EVALUATION_PROTOCOL,
                        "dataset_digest": dataset.digest, "input_digest": state["input_digest"],
                        "code_sha256": run.manifest["runtime"]["code"]["sha256"]}
                    session.add(row)
                    journal = {(a.record.get("stage"), a.record.get("number")): a
                        for a in session.query(EvaluationAttempt).filter_by(answer_id=answer.id)
                        if a.record.get("protocol") == EVALUATION_PROTOCOL}
                    for name, stage in state["stages"].items():
                        for attempt in stage["attempts"]:
                            event = journal.get((name, attempt["number"]))
                            old_status = event.status if event else None
                            if event is None:
                                event = EvaluationAttempt(answer_id=answer.id)
                            event.status = attempt["status"]
                            event.record = copy.deepcopy({**attempt, "stage": name, "protocol": EVALUATION_PROTOCOL})
                            session.add(event)
                            if event.status == "error" and old_status != "error":
                                print(f"Evaluation stage failed: {answer.id} [{name}] attempt {attempt['number']} ({attempt['error_type']})")
            q = questions[answer.question_id]
            result = evaluator(q, PipelineResult.model_validate(answer.payload), [chunks[cid] for cid in q.reference_chunk_ids],
                previous=previous, on_checkpoint=persist, max_attempts=budget, retry_errors=retry_errors)
            persist(result)
            print(f"Evaluated {answer.id}: {evaluation_status(result['metrics'], result['details'])}")
        report = run_status(run_id)
        report["evaluation_complete"] = report["automatically_scored"] == report["expected"]
        return report


def analysis_frames(run_id: int):
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError("Unknown Phase 2 run")
        dataset = session.get(Dataset, run.dataset_id)
        questions = {q["id"]: q for q in dataset.questions}
        answers = {(a.question_id, a.condition): a for a in session.query(Result).filter_by(run_id=run_id)}
        scores = {s.answer_id: s for s in session.query(Score).join(Result, Score.answer_id == Result.id).filter(Result.run_id == run_id)}
        manual = session.query(ManualScore).join(Result, ManualScore.answer_id == Result.id).filter(Result.run_id == run_id).all()
        rows = []
        for item in run.schedule:
            qid, cond = item["question_id"], item["condition"]
            q, answer = questions[qid], answers.get((qid, cond))
            payload = answer.payload if answer and answer.payload else {}
            row = {"answer_id": answer.id if answer else None, "question_id": qid, "condition": cond,
                   "qtype": q["qtype"], "expected_action": q["expected_action"], "is_answerable": q["is_answerable"],
                   "acceptable_actions": q["acceptable_actions"], "family_id": q["family_id"],
                   "split": dataset.split, "status": answer.status if answer else "not_attempted",
                   "action": payload.get("action"), "latency_ms": payload.get("latency_ms"),
                   "attempts": len(answer.attempts) if answer else 0}
            usage = payload.get("usage") or {}
            for field in ("input_tokens", "output_tokens", "total_tokens", "llm_calls"):
                row[field] = usage.get(field)
            score = scores.get(answer.id) if answer else None
            state = evaluation_status(score.metrics, score.details) if score else "not_started"
            row.update(scored=state == "complete", evaluation_recorded=score is not None, evaluation_status=state,
                       evaluation_protocol=(score.details.get("evaluation", {}).get("protocol") if score else None))
            usable = score is not None and state not in {"invalid", "legacy_unversioned"}
            for metric in METRIC_STAGE:
                meta = score.details["evaluation"]["metric_states"][metric] if usable else {}
                status = meta.get("status", "legacy_unversioned" if score else (
                    "generation_failed" if answer and answer.status == "error" else "pending"))
                if state == "invalid":
                    status = "invalid_evaluation"
                row[metric] = score.metrics.get(metric) if usable and status == "ok" else None
                row[metric + "__status"] = status
                row[metric + "__reason"] = meta.get("reason") or (None if status == "ok" else status)
            if usable:
                for name, stage in score.details["evaluation"]["stages"].items():
                    row["stage_" + name + "__status"] = stage["status"]
                    row["stage_" + name + "__attempts"] = len(stage["attempts"])
                    errors = [a for a in stage["attempts"] if a["status"] == "error"]
                    row["stage_" + name + "__last_error"] = errors[-1].get("error") if errors else None
                row.update({"diagnostic_" + name: value for name, value in score.details.get("diagnostic_flags", {}).items()
                            if isinstance(value, bool)})
            rows.append(row)
        human = [{"answer_id": m.answer_id, "evaluator": m.evaluator, "source": m.source, **m.scores}
                 for m in manual if m.source == "human"]
    return pd.DataFrame(rows), pd.DataFrame(human)


def analyze_run(run_id: int, out: Path, allow_incomplete: bool = False):
    from ragapp.phase2.analysis import analyze_scores
    from ragapp.phase2.datasets import digest_json
    import json
    verify_run(run_id, check_live_corpus=False)
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError("Unknown Phase 2 run")
        manifest = run.manifest
    report = run_status(run_id)
    complete = report["successful"] == report["expected"] == report["automatically_scored"]
    if not complete and not allow_incomplete:
        raise ValueError("Complete execution and evaluation first, or explicitly request --allow-incomplete")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Analysis output directory is not empty; use a new versioned directory")
    frame, manual = analysis_frames(run_id)
    out.mkdir(parents=True, exist_ok=True)
    result = analyze_scores(frame, out, manual_df=manual, seed=manifest["seed"])
    with session_scope() as session:
        failures = session.query(EvaluationAttempt).join(Result, EvaluationAttempt.answer_id == Result.id).filter(
            Result.run_id == run_id, EvaluationAttempt.status == "error").all()
        failure_records = [{"answer_id": a.answer_id, **a.record} for a in failures]
    (out / "evaluation_attempt_errors.json").write_text(
        json.dumps(failure_records, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    audit = {"run": report, "complete": complete,
             "interpretation": "held-out within-domain evaluation" if report["split"] == "test" else "development-set exploratory evaluation",
             "frozen_manifest": manifest, "input_data_digest": digest_json(frame.to_json(orient="records")),
             "independent_human_rows": len(manual),
             "evaluation_protocol": manifest.get("evaluation_protocol"),
             "caution": "Quality is conditional on joint applicability and available evaluations. Evaluation failures are not zero scores or N/A; missingness may be selective."}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result
