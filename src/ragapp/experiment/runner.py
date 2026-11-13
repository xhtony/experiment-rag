"""Experiment runner: repeated-measures design.

Every question is answered by every condition within a run. Results are
idempotently persisted to PostgreSQL ((run, condition, question) is unique),
so an interrupted run can simply be resumed.
"""

import time
import traceback

from ragapp.config import get_settings
from ragapp.schemas import AnswerPayload, Condition
from ragapp.pipelines.registry import get_graph
from ragapp.storage.db import session_scope
from ragapp.storage.models import Answer, ExperimentRun, Question


def create_run(name: str) -> int:
    with session_scope() as session:
        run = ExperimentRun(name=name, config=get_settings().snapshot())
        session.add(run)
        session.flush()
        return run.id


def _persist(session, run_id: int, payload: AnswerPayload, latency_ms: int) -> None:
    session.add(
        Answer(
            run_id=run_id,
            question_id=payload.question_id,
            condition=payload.condition.value,
            answer_text=payload.answer_text,
            refused=payload.refused,
            refusal_reason=payload.refusal_reason or "",
            sentences=[s.model_dump() for s in payload.sentences],
            cited_chunk_ids=payload.cited_chunk_ids,
            retrieved_chunk_ids=[c.chunk_id for c in payload.retrieved_chunks],
            latency_ms=latency_ms,
            trace=payload.trace,
        )
    )


def run_experiment(
    run_id: int,
    conditions: list[Condition],
    question_ids: list[int] | None = None,
) -> None:
    with session_scope() as session:
        q = session.query(Question)
        if question_ids:
            q = q.filter(Question.id.in_(question_ids))
        questions = q.order_by(Question.id).all()
        done = {
            (a.condition, a.question_id)
            for a in session.query(Answer).filter(Answer.run_id == run_id).all()
        }

    total = len(questions) * len(conditions)
    print(f"Run {run_id}: {len(questions)} questions x {len(conditions)} conditions "
          f"= {total} answers ({len(done)} already done)")

    for cond in conditions:
        graph = get_graph(cond)
        for question in questions:
            if (cond.value, question.id) in done:
                continue
            t0 = time.perf_counter()
            try:
                final_state = graph.invoke(
                    {"question": question.text, "question_id": question.id}
                )
                payload: AnswerPayload = final_state["payload"]
            except Exception:
                print(f"[{cond.value}] Q{question.id} FAILED:")
                traceback.print_exc()
                continue
            latency_ms = int((time.perf_counter() - t0) * 1000)
            with session_scope() as session:
                _persist(session, run_id, payload, latency_ms)
            status = "REFUSED" if payload.refused else f"{len(payload.answer_text)} chars"
            print(f"[{cond.value}] Q{question.id} ok ({latency_ms} ms, {status})")
