"""Frozen paired schedule, resumable records, and explicit technical failures."""

import random
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import text

from ragapp.config import get_settings
from ragapp.phase2.contracts import PipelineResult, QuestionRecord
from ragapp.phase2.datasets import digest_json
from ragapp.phase2.evaluation_state import EVALUATION_PROTOCOL, evaluation_status
from ragapp.phase2.models import Dataset, Result, Run
from ragapp.phase2.provenance import runtime_manifest, verify_corpus
from ragapp.schemas import Condition
from ragapp.storage.db import get_engine, session_scope


def make_schedule(question_ids: list[int], conditions: list[str], seed: int):
    if not question_ids or len(set(question_ids)) != len(question_ids):
        raise ValueError("Schedule requires unique non-empty question IDs")
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Schedule requires unique non-empty conditions")
    rng = random.Random(seed)
    ids = sorted(question_ids)
    rng.shuffle(ids)
    schedule = []
    for qid in ids:
        order = sorted(conditions)
        rng.shuffle(order)
        schedule.extend({"question_id": qid, "condition": c} for c in order)
    return schedule


def freeze_run(dataset_name: str, name: str, seed: int | None = None,
               conditions: list[str] | None = None) -> int:
    chunks, corpus_digest = verify_corpus()
    settings = get_settings()
    if settings.llm_model == settings.judge_model:
        raise ValueError("Phase 2 protocol requires a judge model distinct from the generator")
    conditions = conditions or [c.value for c in Condition]
    conditions = [Condition(c).value for c in conditions]
    seed = settings.phase2_seed if seed is None else seed
    if seed < 0:
        raise ValueError("The reproducibility seed must be nonnegative")
    with session_scope() as session:
        dataset = session.query(Dataset).filter_by(name=dataset_name).one_or_none()
        if not dataset:
            raise ValueError("Unknown dataset")
        if dataset.corpus_digest != corpus_digest:
            raise ValueError("Dataset gold labels refer to a different corpus version")
        if digest_json(dataset.questions) != dataset.digest:
            raise ValueError("Dataset was modified after registration")
        if dataset.split == "test":
            if set(conditions) != {c.value for c in Condition}:
                raise ValueError("Held-out runs must compare all five conditions")
            if session.query(Run).filter_by(dataset_id=dataset.id).first():
                raise ValueError("Held-out set already has a sealed run; resume it, do not retune/retest")
        from ragapp.phase2.analysis import analysis_plan
        plan = analysis_plan()
        manifest = {"protocol_version": "phase2-v3", "evaluation_protocol": EVALUATION_PROTOCOL, "runtime": runtime_manifest(),
                    "dataset_digest": dataset.digest, "corpus_digest": corpus_digest,
                    "reference_chunks_digest": digest_json(chunks),
                    "split": dataset.split, "conditions": conditions, "seed": seed,
                    "analysis_plan": plan, "unit": "question", "quality_missingness": "pairwise joint applicability"}
        schedule = make_schedule([q["id"] for q in dataset.questions], conditions, seed)
        manifest["schedule_digest"] = digest_json(schedule)
        run = Run(dataset_id=dataset.id, name=name, manifest=manifest, schedule=schedule,
                  reference_chunks=chunks, status="frozen")
        session.add(run)
        session.flush()
        return run.id


def verify_run(run_id: int, check_live_corpus: bool = True):
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError("Unknown Phase 2 run ID")
        dataset = session.get(Dataset, run.dataset_id)
        if digest_json(dataset.questions) != run.manifest["dataset_digest"]:
            raise ValueError("Frozen question labels have changed")
        if digest_json(run.schedule) != run.manifest["schedule_digest"]:
            raise ValueError("Frozen execution schedule has changed")
        if digest_json(run.reference_chunks) != run.manifest["reference_chunks_digest"]:
            raise ValueError("Frozen reference evidence has changed")
        if run.manifest["runtime"] != runtime_manifest():
            raise ValueError("Code, configuration, dependencies, or endpoints changed since freeze; do not mix versions")
    if check_live_corpus:
        chunks, fingerprint = verify_corpus()
        if fingerprint != run.manifest["corpus_digest"] or chunks != run.reference_chunks:
            raise ValueError("Corpus changed since this run was frozen")
    return run, dataset


@contextmanager
def execution_lock(run_id: int):
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        yield
        return
    with engine.connect() as connection:
        acquired = connection.execute(text("SELECT pg_try_advisory_lock(:namespace, :run_id)"),
                                      {"namespace": 901002, "run_id": run_id}).scalar()
        if not acquired:
            raise ValueError("This Phase 2 run is already being executed/evaluated by another process")
        try:
            yield
        finally:
            connection.execute(text("SELECT pg_advisory_unlock(:namespace, :run_id)"),
                               {"namespace": 901002, "run_id": run_id})


def run_experiment(run_id: int, retry_errors: bool = False, pipeline=None):
    if pipeline is None:
        from ragapp.phase2.pipelines import run_pipeline
        pipeline = run_pipeline
    with execution_lock(run_id):
        run, dataset = verify_run(run_id)
        questions = {q["id"]: QuestionRecord.model_validate(q) for q in dataset.questions}
        with session_scope() as session:
            session.get(Run, run_id).status = "running"
        for item in run.schedule:
            qid, condition = item["question_id"], item["condition"]
            with session_scope() as session:
                row = session.query(Result).filter_by(run_id=run_id, question_id=qid, condition=condition).one_or_none()
                if row and (row.status == "success" or (row.status == "error" and not retry_errors)):
                    continue
                attempts = list(row.attempts) if row else []
                row_id = row.id if row else str(uuid.uuid4())
            while len(attempts) < get_settings().phase2_max_attempts:
                started = time.perf_counter()
                entry = {"started_at": datetime.now(timezone.utc).isoformat()}
                # Only question text and ID cross the generation boundary.
                try:
                    payload = pipeline(Condition(condition), qid, questions[qid].question)
                    payload = PipelineResult.model_validate(payload)
                    if payload.condition.value != condition or payload.question_id != qid:
                        raise ValueError("Pipeline returned an incorrect condition/question identity")
                    payload.latency_ms = int((time.perf_counter() - started) * 1000)
                    entry.update(status="success", latency_ms=payload.latency_ms)
                    result_payload, status = payload.model_dump(mode="json"), "success"
                except Exception as exc:
                    entry.update(status="error", error_type=type(exc).__name__,
                                 latency_ms=int((time.perf_counter() - started) * 1000))
                    # Avoid storing SDK exception bodies that may contain credentials or private input.
                    result_payload, status = None, "error"
                attempts.append(entry)
                with session_scope() as session:
                    row = session.get(Result, row_id) or Result(id=row_id, run_id=run_id, question_id=qid, condition=condition)
                    row.status, row.payload, row.attempts = status, result_payload, list(attempts)
                    session.add(row)
                print(f"Phase2 run {run_id}: Q{qid} {condition} {status} (attempt {len(attempts)})")
                if status == "success":
                    break
        report = run_status(run_id)
        with session_scope() as session:
            session.get(Run, run_id).status = "completed" if report["successful"] == report["expected"] else "incomplete"
        return run_status(run_id)


def run_status(run_id: int):
    from ragapp.phase2.models import EvaluationAttempt, Score
    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError("Unknown Phase 2 run")
        dataset = session.get(Dataset, run.dataset_id)
        answers = session.query(Result).filter_by(run_id=run_id).all()
        scores = session.query(Score).join(Result, Score.answer_id == Result.id).filter(Result.run_id == run_id).all()
        statuses = [evaluation_status(s.metrics, s.details) for s in scores]
        evaluation_errors = session.query(EvaluationAttempt).join(Result, EvaluationAttempt.answer_id == Result.id).filter(
            Result.run_id == run_id, EvaluationAttempt.status == "error").all()
        return {"run_id": run_id, "name": run.name, "status": run.status, "split": dataset.split,
                "questions": len(dataset.questions), "conditions": run.manifest["conditions"],
                "expected": len(run.schedule), "successful": sum(a.status == "success" for a in answers),
                "technical_errors": sum(a.status == "error" for a in answers),
                "not_attempted": len(run.schedule) - len(answers), "automatically_scored": statuses.count("complete"),
                "evaluation_records": len(scores), "evaluation_partial": statuses.count("partial"),
                "evaluation_failed": statuses.count("failed"), "evaluation_pending": statuses.count("pending"),
                "evaluation_legacy_or_invalid": sum(s in {"invalid", "legacy_unversioned"} for s in statuses),
                "evaluation_not_started": sum(a.status == "success" for a in answers) - len(scores),
                "evaluation_error_attempts": len(evaluation_errors),
                "evaluation_error_types": sorted({e.record.get("error_type", "") for e in evaluation_errors}),
                "error_types": sorted({t.get("error_type", "") for a in answers for t in a.attempts if t["status"] == "error"})}
