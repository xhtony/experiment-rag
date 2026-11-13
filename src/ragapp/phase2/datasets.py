"""Immutable question datasets and explicit development/test separation."""

import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from ragapp.phase2.contracts import QuestionRecord
from ragapp.phase2.models import Dataset
from ragapp.storage.db import session_scope


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_questions(path: Path) -> list[QuestionRecord]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                rows.append(QuestionRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid question at line {number}: {exc}") from exc
    if not rows:
        raise ValueError("Question file is empty")
    if len({q.id for q in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    normalized = [normalize(q.question) for q in rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate question texts")
    return sorted(rows, key=lambda q: q.id)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def overlap_report(questions, development):
    issues = []
    for question in questions:
        for dev in development:
            same_family = bool(question.family_id and question.family_id == dev.family_id)
            similarity = SequenceMatcher(None, normalize(question.question), normalize(dev.question)).ratio()
            if same_family or similarity >= 0.9:
                issues.append({"question_id": question.id, "development_id": dev.id,
                               "same_family": same_family, "text_similarity": round(similarity, 4)})
    return issues


def validate_questions(questions, chunk_ids: set[str], split: str):
    if split not in {"development", "test"}:
        raise ValueError("split must be development or test")
    missing = {cid for q in questions for cid in q.reference_chunk_ids if cid not in chunk_ids}
    if missing:
        raise ValueError(f"Unknown reference chunk IDs (target gold or background): {sorted(missing)}")
    if split == "test" and any(not q.gold_reviewed for q in questions):
        raise ValueError("Every held-out test question requires gold_reviewed=true after human audit")


def import_dataset(path: Path, name: str, split: str, chunks: list[dict], corpus_digest: str,
                   development_file: Path | None = None, attest_heldout: bool = False,
                   overlap_review: Path | None = None) -> int:
    questions = read_questions(path)
    validate_questions(questions, {c["chunk_id"] for c in chunks}, split)
    provenance = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "qtype_counts": dict(Counter(q.qtype.value for q in questions)),
                  "attest_heldout": attest_heldout}
    with session_scope() as session:
        if session.query(Dataset).filter_by(name=name).first():
            raise ValueError("Dataset names are immutable; use a new versioned name")
        other = session.query(Dataset).all()
        if split == "test":
            if not attest_heldout or development_file is None:
                raise ValueError("Test import requires --development-file and --attest-heldout")
            # The raw Phase 1 file is valid for overlap screening, not for gold approval.
            old = [json.loads(line) for line in development_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            development = [QuestionRecord(id=r["id"], question=r["question"], qtype=r["qtype"],
                           is_answerable=r.get("is_answerable", True), expected_action="ANSWER",
                           family_id=r.get("family_id", "")) for r in old]
            development += [QuestionRecord.model_validate(q) for ds in other for q in ds.questions]
            issues = overlap_report(questions, development)
            within = []
            for i, q in enumerate(questions):
                within += overlap_report([q], questions[:i])
            provenance["overlap_flags"] = issues
            provenance["within_test_similarity_flags"] = within
            provenance["development_sha256"] = hashlib.sha256(development_file.read_bytes()).hexdigest()
            if issues:
                if any(i["same_family"] or i["text_similarity"] == 1.0 for i in issues):
                    raise ValueError(f"Held-out question/family overlaps previous datasets: {issues[:8]}")
                review = json.loads(overlap_review.read_text(encoding="utf-8")) if overlap_review else {}
                if not review.get("reviewer") or not review.get("rationale") or review.get("flags_digest") != digest_json(issues):
                    raise ValueError(f"Near-duplicate review required; flags_digest={digest_json(issues)}; flags={issues[:8]}")
                provenance["overlap_review"] = review
            if within:
                raise ValueError(f"Test questions must use distinct families and avoid near duplicates: {within[:8]}")
        else:
            tests = [QuestionRecord.model_validate(q) for ds in other if ds.split == "test" for q in ds.questions]
            if overlap_report(questions, tests):
                raise ValueError("Development import overlaps an already sealed test set")
        serialized = [q.model_dump(mode="json") for q in questions]
        dataset = Dataset(name=name, split=split, digest=digest_json(serialized), corpus_digest=corpus_digest,
                          questions=serialized, provenance=provenance)
        session.add(dataset)
        session.flush()
        return dataset.id


def prepare_development(source: Path, out: Path) -> int:
    """Convert legacy rows to editable drafts, never pretend labels were human audited."""
    if out.exists():
        raise ValueError("Output exists; refusing to overwrite question annotations")
    rows = []
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        action = "CLARIFY" if row["qtype"] == "ambiguous" else "REFUSE" if row.get("expect_refusal") else "ANSWER"
        record = QuestionRecord(id=row["id"], question=row["question"], qtype=row["qtype"],
            is_answerable=row.get("is_answerable", True), expected_action=action,
            gold_answer=row.get("gold_answer", ""), gold_chunk_ids=row.get("gold_chunk_ids", []),
            family_id=f"legacy-{row['id']}", gold_reviewed=False)
        rows.append(record.model_dump(mode="json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return len(rows)
