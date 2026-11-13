"""Database schema.

Design rationale (maps onto the methodology):
- documents / chunks      -> controlled corpus with full provenance metadata
- questions               -> gold-standard question dataset (5 types)
- experiment_runs         -> one row per experiment execution, with frozen config
- answers                 -> one row per (run, condition, question), full trace
- auto_scores             -> unified automated evaluation (LLM-as-judge)
- manual_scores           -> human rubric scores, supports inter-rater analysis
"""

import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # slug
    title: Mapped[str] = mapped_column(String(512))
    version: Mapped[str] = mapped_column(String(64), default="")
    doc_date: Mapped[str] = mapped_column(String(32), default="")
    source_path: Mapped[str] = mapped_column(String(1024), default="")
    source_url: Mapped[str] = mapped_column(String(1024), default="")

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)  # "<doc>::<seq>"
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    seq: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(512), default="")
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="chunks")


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    qtype: Mapped[str] = mapped_column(String(32))  # QuestionType
    is_answerable: Mapped[bool] = mapped_column(Boolean, default=True)
    expect_refusal: Mapped[bool] = mapped_column(Boolean, default=False)
    gold_answer: Mapped[str] = mapped_column(Text, default="")
    gold_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # frozen Settings snapshot


class Answer(Base):
    __tablename__ = "answers"
    __table_args__ = (
        UniqueConstraint("run_id", "condition", "question_id", name="uq_answer"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id"))
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"))
    condition: Mapped[str] = mapped_column(String(8))  # C0..C3

    answer_text: Mapped[str] = mapped_column(Text, default="")
    refused: Mapped[bool] = mapped_column(Boolean, default=False)
    refusal_reason: Mapped[str] = mapped_column(Text, default="")
    sentences: Mapped[list] = mapped_column(JSON, default=list)        # CitedSentence[]
    cited_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    trace: Mapped[dict] = mapped_column(JSON, default=dict)


class AutoScore(Base):
    __tablename__ = "auto_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"), unique=True)

    # Faithfulness (claim-level, judged against retrieved context;
    # for C0 judged against gold evidence chunks)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)  # 0..1
    total_claims: Mapped[int] = mapped_column(Integer, default=0)
    unsupported_claims: Mapped[int] = mapped_column(Integer, default=0)
    contradicted_claims: Mapped[int] = mapped_column(Integer, default=0)

    # Hallucination: >=1 unsupported/fabricated claim in a non-refused answer
    hallucinated: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Citation accuracy vs gold supporting chunks
    citation_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    citation_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Sentence-level: fraction of citations that actually support their sentence
    citation_support: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Refusal behaviour (confusion-matrix cell for this answer)
    refusal_outcome: Mapped[str] = mapped_column(String(32), default="")
    # one of: correct_answer | correct_refusal | false_refusal | false_answer

    # Retrieval quality (diagnostic, supports error analysis)
    retrieval_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    retrieval_recall: Mapped[float | None] = mapped_column(Float, nullable=True)

    details: Mapped[dict] = mapped_column(JSON, default=dict)  # claim verdicts etc.


class ManualScore(Base):
    __tablename__ = "manual_scores"
    __table_args__ = (
        UniqueConstraint("answer_id", "evaluator", name="uq_manual_score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"))
    evaluator: Mapped[str] = mapped_column(String(128))

    # Rubric: 2 = fully supported, 1 = partially supported, 0 = unsupported
    faithfulness_rubric: Mapped[int | None] = mapped_column(Integer, nullable=True)
    citation_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hallucinated: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    refusal_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    error_category: Mapped[str] = mapped_column(String(64), default="")
    # retrieval_failure | ranking_failure | generation_failure | citation_failure |
    # over_answering | over_refusal | policy_interpretation_error | none
    notes: Mapped[str] = mapped_column(Text, default="")
