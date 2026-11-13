"""Explicit study labels, distinct from the question-only generation interface."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragapp.schemas import CitedSentence, Condition, QuestionType, RetrievedChunk


class Action(str, Enum):
    ANSWER = "ANSWER"
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"


class QuestionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    question: str = Field(min_length=1)
    qtype: QuestionType
    is_answerable: bool
    expected_action: Action
    acceptable_actions: list[Action] = Field(default_factory=list)
    gold_answer: str = ""
    gold_chunk_ids: list[str] = Field(default_factory=list)
    background_chunk_ids: list[str] = Field(default_factory=list,
        description="Reference-only evidence for contextual facts; never a target retrieval gold label.")
    gold_evidence_groups: list[list[str]] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    acceptable_clarifications: list[str] = Field(default_factory=list)
    family_id: str = ""
    annotation_notes: str = ""
    gold_reviewed: bool = False

    @property
    def reference_chunk_ids(self) -> list[str]:
        return self.gold_chunk_ids + self.background_chunk_ids

    @model_validator(mode="after")
    def check_labels(self):
        self.question = self.question.strip()
        if not self.question:
            raise ValueError("question cannot be blank")
        if not self.acceptable_actions:
            self.acceptable_actions = [self.expected_action]
        if self.expected_action not in self.acceptable_actions:
            raise ValueError("expected_action must be an acceptable action")
        if len(set(self.acceptable_actions)) != len(self.acceptable_actions):
            raise ValueError("acceptable_actions must be unique")
        if len(set(self.gold_chunk_ids)) != len(self.gold_chunk_ids):
            raise ValueError("gold_chunk_ids must be unique")
        if len(set(self.background_chunk_ids)) != len(self.background_chunk_ids):
            raise ValueError("background_chunk_ids must be unique")
        if set(self.gold_chunk_ids) & set(self.background_chunk_ids):
            raise ValueError("Target gold and background evidence must be disjoint")
        for values in (self.gold_chunk_ids, self.background_chunk_ids, self.required_facts, self.missing_slots, self.acceptable_clarifications):
            if any(not value.strip() for value in values):
                raise ValueError("Annotation lists cannot contain blank entries")
        if any(not group for group in self.gold_evidence_groups):
            raise ValueError("gold evidence groups cannot be empty")
        if any(cid not in self.gold_chunk_ids for group in self.gold_evidence_groups for cid in group):
            raise ValueError("every alternative evidence ID must be in gold_chunk_ids")
        if self.gold_reviewed:
            if not self.family_id.strip() or not self.annotation_notes.strip():
                raise ValueError("reviewed labels need family_id and annotation_notes")
            if Action.ANSWER in self.acceptable_actions and not (
                self.gold_answer.strip() and self.gold_chunk_ids and self.required_facts
            ):
                raise ValueError("answerable gold needs answer, evidence and required facts")
            if self.expected_action == Action.CLARIFY and not self.missing_slots:
                raise ValueError("clarification gold needs missing_slots")
            if self.expected_action == Action.ANSWER and not self.is_answerable:
                raise ValueError("ANSWER conflicts with is_answerable=false")
        return self


class PipelineResult(BaseModel):
    condition: Condition
    question_id: int
    action: Action
    answer_text: str = ""
    reason: str = ""
    clarification_question: str = ""
    sentences: list[CitedSentence] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    trace: dict = Field(default_factory=dict)
    usage: dict = Field(default_factory=dict)
    latency_ms: int = 0
