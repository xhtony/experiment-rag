"""Versioned Phase 2 records. Never migrate or overwrite Phase 1 result tables."""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow():
    return datetime.now(timezone.utc)


class Phase2Base(DeclarativeBase):
    pass


class CorpusState(Phase2Base):
    __tablename__ = "phase2_corpus_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    manifest: Mapped[dict] = mapped_column(JSON)


class Dataset(Phase2Base):
    __tablename__ = "phase2_datasets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    split: Mapped[str] = mapped_column(String(20))
    digest: Mapped[str] = mapped_column(String(64))
    corpus_digest: Mapped[str] = mapped_column(String(64))
    questions: Mapped[list] = mapped_column(JSON)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Run(Phase2Base):
    __tablename__ = "phase2_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("phase2_datasets.id"))
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(32), default="frozen")
    manifest: Mapped[dict] = mapped_column(JSON)
    schedule: Mapped[list] = mapped_column(JSON)
    reference_chunks: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Result(Phase2Base):
    __tablename__ = "phase2_answers"
    __table_args__ = (UniqueConstraint("run_id", "question_id", "condition", name="uq_phase2_answer"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("phase2_runs.id"))
    question_id: Mapped[int] = mapped_column(Integer)
    condition: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(20))
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attempts: Mapped[list] = mapped_column(JSON, default=list)


class Score(Phase2Base):
    __tablename__ = "phase2_auto_scores"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    answer_id: Mapped[str] = mapped_column(ForeignKey("phase2_answers.id"), unique=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)


class EvaluationAttempt(Phase2Base):
    __tablename__ = "phase2_evaluation_attempts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    answer_id: Mapped[str] = mapped_column(ForeignKey("phase2_answers.id"))
    status: Mapped[str] = mapped_column(String(20))
    record: Mapped[dict] = mapped_column(JSON)


class ManualBatch(Phase2Base):
    __tablename__ = "phase2_manual_batches"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("phase2_runs.id"))
    seed: Mapped[int] = mapped_column(Integer)
    records: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ManualScore(Phase2Base):
    __tablename__ = "phase2_manual_scores"
    __table_args__ = (UniqueConstraint("answer_id", "evaluator", name="uq_phase2_manual"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    answer_id: Mapped[str] = mapped_column(ForeignKey("phase2_answers.id"))
    evaluator: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32), default="human")
    scores: Mapped[dict] = mapped_column(JSON)
    batch_id: Mapped[str] = mapped_column(ForeignKey("phase2_manual_batches.id"))


def init_phase2_db():
    from ragapp.storage.db import get_engine
    Phase2Base.metadata.create_all(get_engine())
