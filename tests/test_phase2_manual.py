import csv
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragapp.phase2.contracts import PipelineResult, QuestionRecord
from ragapp.phase2.datasets import digest_json
from ragapp.phase2.manual import export_manual, import_manual, select_questions
from ragapp.phase2.models import Dataset, ManualBatch, ManualScore, Phase2Base, Result, Run
from ragapp.schemas import RetrievedChunk
from ragapp.storage import db


class ManualTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "packet.csv"
        self.engine = create_engine("sqlite:///:memory:")
        Phase2Base.metadata.create_all(self.engine)
        self.old_engine, self.old_session = db._engine, db._SessionLocal
        db._engine, db._SessionLocal = self.engine, sessionmaker(bind=self.engine, expire_on_commit=False)
        self.chunk = RetrievedChunk(chunk_id="a", document_title="Policy", text="Applications close Monday.").model_dump()
        self.questions = []
        for qid, qtype, action in [(1, "single_doc", "ANSWER"), (2, "ambiguous", "CLARIFY"), (3, "unanswerable", "REFUSE")]:
            q = QuestionRecord(id=qid, question="=1+1" if qid == 1 else f"Question {qid}", qtype=qtype,
                is_answerable=action == "ANSWER", expected_action=action, gold_answer="Monday" if action == "ANSWER" else "",
                gold_chunk_ids=["a"], required_facts=["Monday"] if action == "ANSWER" else [],
                missing_slots=["programme"] if action == "CLARIFY" else [], family_id=f"f{qid}",
                annotation_notes="Reviewed.", gold_reviewed=True)
            self.questions.append(q.model_dump(mode="json"))
        self.conditions = ["C0", "C1", "C2", "C3", "C4"]
        self.schedule = [{"question_id": q["id"], "condition": c} for q in self.questions for c in self.conditions]
        with db.session_scope() as session:
            dataset = Dataset(name="manual-fixture", split="test", digest=digest_json(self.questions),
                              corpus_digest="fixture", questions=self.questions, provenance={})
            session.add(dataset)
            session.flush()
            run = Run(dataset_id=dataset.id, name="manual-test", status="completed", schedule=self.schedule,
                      reference_chunks=[self.chunk], manifest={"conditions": self.conditions})
            session.add(run)
            session.flush()
            self.run_id = run.id
            for item in self.schedule:
                q = self.questions[item["question_id"] - 1]
                payload = PipelineResult(condition=item["condition"], question_id=q["id"], action=q["expected_action"],
                    answer_text="Monday." if q["expected_action"] == "ANSWER" else "",
                    reason="No evidence." if q["expected_action"] == "REFUSE" else "",
                    clarification_question="Which programme?" if q["expected_action"] == "CLARIFY" else "",
                    retrieved_chunks=[] if item["condition"] == "C0" else [self.chunk]).model_dump(mode="json")
                session.add(Result(id=f"private-{q['id']}-{item['condition']}", run_id=run.id, question_id=q["id"],
                                   condition=item["condition"], status="success", payload=payload, attempts=[]))

    def tearDown(self):
        db._engine, db._SessionLocal = self.old_engine, self.old_session
        self.engine.dispose()
        self.temp.cleanup()

    def read(self):
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def write(self, rows):
        with self.path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_export_is_blinded_and_never_prefills_scores(self):
        result = export_manual(self.run_id, self.path)
        rows = self.read()
        self.assertEqual(result["rows"], 15)
        self.assertNotIn("condition", rows[0])
        self.assertNotIn("answer_id", rows[0])
        self.assertTrue(all(not r["reference_groundedness"] for r in rows))
        self.assertNotIn("private-", self.path.read_text(encoding="utf-8-sig"))
        self.assertTrue(any(r["question"] == "'=1+1" for r in rows))

    def test_refusal_only_scores_import_without_faithfulness(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        row = next(r for r in rows if r["action"] == "REFUSE")
        row["action_appropriate"] = "1"
        self.write(rows)
        self.assertEqual(import_manual(self.path, "researcher", attest_human=True), 1)
        self.assertEqual(import_manual(self.path, "researcher", attest_human=True), 0)
        with db.session_scope() as session:
            score = session.query(ManualScore).one()
            self.assertIsNone(score.scores["reference_groundedness"])
            self.assertEqual(score.source, "human")

    def test_human_attestation_required(self):
        export_manual(self.run_id, self.path)
        with self.assertRaisesRegex(ValueError, "attest-human"):
            import_manual(self.path, "rater")

    def test_background_evidence_is_separate_and_coverage_is_blank(self):
        with db.session_scope() as session:
            dataset = session.query(Dataset).one()
            questions = [dict(q) for q in dataset.questions]
            questions[2]["gold_chunk_ids"] = []
            questions[2]["background_chunk_ids"] = ["a"]
            dataset.questions = questions
        export_manual(self.run_id, self.path)
        rows = self.read()
        row = next(r for r in rows if r["action"] == "REFUSE")
        self.assertEqual(row["reference_evidence"], "")
        self.assertIn("Applications close Monday", row["background_reference_evidence"])
        self.assertEqual(row["required_fact_coverage_applicable"], "False")
        self.assertTrue(all(r["required_fact_coverage"] == "" for r in rows))

    def test_manual_required_fact_coverage_import(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        row = next(r for r in rows if r["action"] == "ANSWER")
        self.assertEqual(row["required_fact_coverage_applicable"], "True")
        row["required_fact_coverage"] = "0.5"
        self.write(rows)
        self.assertEqual(import_manual(self.path, "fixture-human", attest_human=True), 1)
        with db.session_scope() as session:
            self.assertEqual(session.query(ManualScore).one().scores["required_fact_coverage"], .5)

    def test_manual_coverage_not_applicable_to_unanswerable_question(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        next(r for r in rows if r["action"] == "REFUSE")["required_fact_coverage"] = "0"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "required_fact_coverage is N/A"):
            import_manual(self.path, "fixture-human", attest_human=True)

    def test_manual_citations_not_applicable_to_refusal(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        next(r for r in rows if r["action"] == "REFUSE")["citation_precision"] = "0"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "no .*citation metrics"):
            import_manual(self.path, "fixture-human", attest_human=True)

    def test_ai_assisted_ratings_remain_marked(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        rows[0]["action_appropriate"] = "1"
        self.write(rows)
        self.assertEqual(import_manual(self.path, "assisted", "ai_assisted"), 1)
        with db.session_scope() as session:
            self.assertEqual(session.query(ManualScore).one().source, "ai_assisted")

    def test_changed_packet_fields_rejected_atomically(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        rows[0]["action_appropriate"] = "1"
        rows[-1]["answer_text"] = "Altered answer"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "Non-rubric"):
            import_manual(self.path, "rater", attest_human=True)
        with db.session_scope() as session:
            self.assertEqual(session.query(ManualScore).count(), 0)

    def test_out_of_range_values_rejected(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        rows[0]["action_appropriate"] = "2"
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "within"):
            import_manual(self.path, "rater", attest_human=True)

    def test_stratified_sample_selects_questions_not_individual_answers(self):
        for sample in (1, 2):
            with self.assertRaises(ValueError):
                select_questions(self.questions, sample, 42)
        self.assertEqual(set(select_questions(self.questions, 3, 42)), {1, 2, 3})

    def test_incomplete_run_not_exported(self):
        with db.session_scope() as session:
            answer = session.query(Result).first()
            answer.status = "error"
        with self.assertRaisesRegex(ValueError, "complete paired"):
            export_manual(self.run_id, self.path)

    def test_duplicate_review_ids_are_rejected(self):
        export_manual(self.run_id, self.path)
        rows = self.read()
        rows.append(rows[0])
        self.write(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            import_manual(self.path, "rater", attest_human=True)


if __name__ == "__main__":
    unittest.main()
