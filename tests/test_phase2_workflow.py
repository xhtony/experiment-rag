"""Offline workflow regression tests: no network, credentials, or production data."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ragapp.config import get_settings
from ragapp.phase2.contracts import PipelineResult, QuestionRecord
from ragapp.phase2.datasets import digest_json, import_dataset, overlap_report, prepare_development
from ragapp.phase2.models import Dataset, EvaluationAttempt, Phase2Base, Result, Run, Score
from ragapp.phase2.runner import freeze_run, make_schedule, run_experiment, run_status, verify_run
from ragapp.schemas import CitedSentence, RetrievedChunk
from ragapp.storage import db
from ragapp.storage.models import Base


class Phase2WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"RAGAPP_ENV_FILE": str(self.root / "absent"),
            "RAGAPP_DATABASE_URL": "sqlite:///:memory:", "RAGAPP_PHASE2_MAX_ATTEMPTS": "2"})
        self.env.start()
        get_settings.cache_clear()
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        Phase2Base.metadata.create_all(self.engine)
        self.old_engine, self.old_session = db._engine, db._SessionLocal
        db._engine, db._SessionLocal = self.engine, sessionmaker(bind=self.engine, expire_on_commit=False)
        self.chunks = [RetrievedChunk(chunk_id="policy::0000", document_title="Policy", text="Applications close Monday.").model_dump()]
        self.corpus = patch("ragapp.phase2.runner.verify_corpus", return_value=(self.chunks, "corpus-1"))
        self.runtime = patch("ragapp.phase2.runner.runtime_manifest", return_value={"code": {"sha256": "code-1"}})
        self.corpus.start()
        self.runtime.start()

    def tearDown(self):
        self.corpus.stop()
        self.runtime.stop()
        db._engine, db._SessionLocal = self.old_engine, self.old_session
        self.engine.dispose()
        self.env.stop()
        get_settings.cache_clear()
        self.temp.cleanup()

    def question(self, qid=1, **updates):
        row = dict(id=qid, question=f"When does application period {qid} close?", qtype="single_doc",
                   is_answerable=True, expected_action="ANSWER", gold_answer="Monday",
                   gold_chunk_ids=["policy::0000"], required_facts=["Monday"], family_id=f"family-{qid}",
                   annotation_notes="Reviewed against the frozen policy.", gold_reviewed=True)
        row.update(updates)
        return QuestionRecord(**row)

    def dataset(self, split="development", questions=None):
        rows = [q.model_dump(mode="json") for q in (questions or [self.question()])]
        with db.session_scope() as session:
            session.add(Dataset(name="questions-v1", split=split, digest=digest_json(rows),
                                corpus_digest="corpus-1", questions=rows, provenance={"attest_heldout": True}))

    def pipeline(self, condition, qid, text):
        self.assertIsInstance(text, str)
        return PipelineResult(condition=condition, question_id=qid, action="ANSWER", answer_text="Monday",
                              sentences=[CitedSentence(text="Monday", citations=["policy::0000"])],
                              retrieved_chunks=self.chunks if condition.value != "C0" else [])

    def test_schedule_is_paired_reproducible_and_interleaved(self):
        schedule = make_schedule([1, 2, 3], ["C0", "C1", "C2", "C3", "C4"], 7)
        self.assertEqual(schedule, make_schedule([3, 2, 1], ["C4", "C3", "C2", "C1", "C0"], 7))
        self.assertEqual(len(schedule), 15)
        for offset in range(0, 15, 5):
            group = schedule[offset:offset + 5]
            self.assertEqual(len({r["question_id"] for r in group}), 1)
            self.assertEqual(len({r["condition"] for r in group}), 5)

    def test_run_and_resume_do_not_duplicate_answers(self):
        self.dataset()
        run_id = freeze_run("questions-v1", "offline")
        report = run_experiment(run_id, pipeline=self.pipeline)
        self.assertEqual(report["successful"], 5)
        with patch.object(self, "pipeline", side_effect=AssertionError("must not rerun completed answers")) as fake:
            run_experiment(run_id, pipeline=fake)
        with db.session_scope() as session:
            self.assertEqual(session.query(Result).count(), 5)

    def test_technical_error_is_not_refusal(self):
        self.dataset()
        run_id = freeze_run("questions-v1", "failure", conditions=["C4"])
        def failure(*args):
            raise TimeoutError("private diagnostic must not be persisted")
        report = run_experiment(run_id, pipeline=failure)
        self.assertEqual(report["technical_errors"], 1)
        self.assertEqual(report["successful"], 0)
        with db.session_scope() as session:
            result = session.query(Result).one()
            self.assertIsNone(result.payload)
            self.assertEqual(len(result.attempts), 2)
            self.assertNotIn("private diagnostic", json.dumps(result.attempts))

    def test_frozen_question_mutation_rejected(self):
        self.dataset()
        run_id = freeze_run("questions-v1", "frozen")
        with db.session_scope() as session:
            dataset = session.query(Dataset).one()
            changed = [dict(dataset.questions[0], gold_answer="Tuesday")]
            dataset.questions = changed
        with self.assertRaisesRegex(ValueError, "labels have changed"):
            verify_run(run_id, check_live_corpus=False)

    def test_repeated_heldout_freeze_rejected(self):
        self.dataset(split="test")
        freeze_run("questions-v1", "test")
        with self.assertRaisesRegex(ValueError, "already has a sealed run"):
            freeze_run("questions-v1", "test-again")

    def test_frozen_reference_mutation_rejected_without_live_services(self):
        self.dataset()
        run_id = freeze_run("questions-v1", "evidence")
        with db.session_scope() as session:
            run = session.get(Run, run_id)
            run.reference_chunks = [dict(run.reference_chunks[0], text="A changed reference.")]
        with self.assertRaisesRegex(ValueError, "reference evidence has changed"):
            verify_run(run_id, check_live_corpus=False)

    def test_heldout_requires_all_conditions(self):
        self.dataset(split="test")
        with self.assertRaisesRegex(ValueError, "all five"):
            freeze_run("questions-v1", "only-c4", conditions=["C4"])

    def test_question_only_drafts_are_not_reviewed(self):
        source = self.root / "legacy.jsonl"
        source.write_text(json.dumps({"id": 1, "question": "Which policy applies?", "qtype": "ambiguous"}), encoding="utf-8")
        out = self.root / "draft.jsonl"
        self.assertEqual(prepare_development(source, out), 1)
        row = json.loads(out.read_text(encoding="utf-8"))
        self.assertFalse(row["gold_reviewed"])
        self.assertEqual(row["expected_action"], "CLARIFY")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_development(source, out)

    def test_heldout_duplicate_of_legacy_is_rejected(self):
        q = self.question()
        test_file, legacy = self.root / "test.jsonl", self.root / "legacy.jsonl"
        test_file.write_text(q.model_dump_json(), encoding="utf-8")
        legacy.write_text(json.dumps({"id": 9, "question": q.question, "qtype": "single_doc"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            import_dataset(test_file, "test", "test", self.chunks, "corpus-1", legacy, True)

    def test_family_overlap_detected_even_when_wording_differs(self):
        a = self.question(question="What documents are required for enrolment?", family_id="same")
        b = self.question(2, question="Tell me the paperwork needed to register.", family_id="same")
        self.assertTrue(overlap_report([a], [b])[0]["same_family"])

    def test_complete_offline_evaluation_persistence(self):
        from ragapp.phase2.workflow import evaluate_run, analysis_frames
        from ragapp.phase2.evaluation import evaluate_answer
        from test_phase2_evaluation import FixtureJudge
        self.dataset()
        run_id = freeze_run("questions-v1", "scoring")
        run_experiment(run_id, pipeline=self.pipeline)
        def score(question, answer, references, **kwargs):
            self.assertEqual(references[0].chunk_id, "policy::0000")
            return evaluate_answer(question, answer, references, FixtureJudge(), **kwargs)
        report = evaluate_run(run_id, evaluator=score)
        self.assertEqual(report["automatically_scored"], 5)
        frame, manual = analysis_frames(run_id)
        self.assertEqual(len(frame), 5)
        self.assertTrue(manual.empty)

    def test_unreviewed_development_cannot_be_scored(self):
        from ragapp.phase2.workflow import evaluate_run
        self.dataset(questions=[self.question(gold_reviewed=False)])
        run_id = freeze_run("questions-v1", "draft")
        run_experiment(run_id, pipeline=self.pipeline)
        with self.assertRaisesRegex(ValueError, "human-reviewed"):
            evaluate_run(run_id, evaluator=lambda *args: {})

    def test_background_references_reach_evaluator_not_generation(self):
        from ragapp.phase2.workflow import evaluate_run, analysis_frames
        from ragapp.phase2.evaluation import evaluate_answer
        from test_phase2_evaluation import FixtureJudge
        question = self.question(qtype="unanswerable", is_answerable=False, expected_action="REFUSE",
            question="What exact application quota is guaranteed?", gold_answer="Not established.",
            gold_chunk_ids=[], background_chunk_ids=["policy::0000"], required_facts=[])
        self.dataset(questions=[question])
        run_id = freeze_run("questions-v1", "background-fixture")
        def pipeline(condition, qid, text):
            self.assertEqual(text, question.question)
            return PipelineResult(condition=condition, question_id=qid, action="REFUSE",
                reason="Applications close Monday, but the quota is not established.",
                retrieved_chunks=self.chunks if condition.value != "C0" else [])
        run_experiment(run_id, pipeline=pipeline)
        def score(q, answer, references, **kwargs):
            self.assertFalse(q.gold_chunk_ids)
            self.assertEqual([c.chunk_id for c in references], q.background_chunk_ids)
            judge = FixtureJudge()
            judge.decompose = lambda a: [{"text": "Applications close Monday.", "sentence_index": -1}]
            return evaluate_answer(q, answer, references, judge, **kwargs)
        report = evaluate_run(run_id, evaluator=score)
        self.assertEqual(report["automatically_scored"], 5)
        frame, _ = analysis_frames(run_id)
        self.assertTrue((frame.reference_groundedness == 1).all())
        self.assertTrue(frame.citation_precision.isna().all())
        self.assertTrue(frame.retrieval_recall.isna().all())

    def test_failed_evaluation_is_recorded_and_can_retry(self):
        from ragapp.phase2.workflow import evaluate_run, analyze_run, analysis_frames
        from ragapp.phase2.evaluation import evaluate_answer
        from test_phase2_evaluation import TrackingJudge
        self.dataset()
        run_id = freeze_run("questions-v1", "retry-eval")
        run_experiment(run_id, pipeline=self.pipeline)
        failing = {"on": True}
        def score(question, answer, references, **kwargs):
            return evaluate_answer(question, answer, references,
                TrackingJudge({"reference_grounding": 10} if failing["on"] else {}), **kwargs)
        first = evaluate_run(run_id, evaluator=score)
        self.assertFalse(first["evaluation_complete"])
        self.assertEqual(first["automatically_scored"], 0)
        self.assertEqual(first["evaluation_records"], 5)
        self.assertEqual(first["evaluation_partial"], 5)
        with db.session_scope() as session:
            failed = session.query(EvaluationAttempt).filter_by(status="error").first()
            self.assertEqual(failed.record["stage"], "reference_grounding")
            self.assertEqual(failed.record["number"], 1)
            self.assertEqual(session.query(Score).count(), 5)
            self.assertEqual(session.query(EvaluationAttempt).filter_by(status="error").count(), 10)
        frame, _ = analysis_frames(run_id)
        self.assertTrue(frame.action_appropriate.all())
        self.assertTrue(frame.reference_groundedness.isna().all())
        self.assertTrue((frame.reference_groundedness__status == "evaluation_failed").all())
        with self.assertRaisesRegex(ValueError, "Complete execution"):
            analyze_run(run_id, self.root / "not-complete")
        analyze_run(run_id, self.root / "partial", allow_incomplete=True)
        self.assertTrue((self.root / "partial" / "metric_availability.csv").exists())
        self.assertTrue((self.root / "partial" / "evaluation_failures.csv").exists())
        self.assertTrue((self.root / "partial" / "evaluation_attempt_errors.json").exists())
        audit = json.loads((self.root / "partial" / "audit.json").read_text())
        self.assertFalse(audit["complete"])
        failing["on"] = False
        # Simply rerunning cannot replenish an exhausted stage budget.
        same = evaluate_run(run_id, evaluator=score)
        self.assertEqual(same["automatically_scored"], 0)
        second = evaluate_run(run_id, evaluator=score, retry_errors=True)
        self.assertTrue(second["evaluation_complete"])
        self.assertEqual(second["automatically_scored"], 5)
        with db.session_scope() as session:
            self.assertEqual(session.query(Score).count(), 5)
            attempts = session.query(EvaluationAttempt).all()
            claims = [a for a in attempts if a.record["stage"] == "claims"]
            self.assertEqual(len(claims), 5)
            self.assertTrue(all(a.record["number"] == 1 for a in claims))
            self.assertEqual(sum(a.status == "error" for a in attempts), 10)
        evaluate_run(run_id, evaluator=lambda *a, **k: self.fail("completed score must not rerun"))

    def test_quote_failure_checkpoint_keeps_diagnostic_and_unrelated_scores(self):
        from ragapp.phase2.workflow import evaluate_run
        from ragapp.phase2.evaluation import evaluate_answer
        from test_phase2_evaluation import FixtureJudge
        self.dataset()
        run_id = freeze_run("questions-v1", "bad-quote", conditions=["C4"])
        run_experiment(run_id, pipeline=self.pipeline)
        judge = FixtureJudge()
        original = judge.verify
        def wrong_quote(requests, passages, purpose):
            values = original(requests, passages, purpose)
            if purpose == "reference_grounding":
                values[0]["evidence_quotes"] = ["An invented deadline."]
            return values
        judge.verify = wrong_quote
        report = evaluate_run(run_id, evaluator=lambda q, a, c, **kw: evaluate_answer(q, a, c, judge, **kw))
        self.assertEqual(report["evaluation_partial"], 1)
        with db.session_scope() as session:
            row = session.query(Score).one()
            self.assertIsNone(row.metrics["reference_groundedness"])
            self.assertEqual(row.metrics["citation_completeness"], 1)
            self.assertTrue(row.metrics["action_appropriate"])
            attempts = session.query(EvaluationAttempt).filter_by(status="error").all()
            self.assertEqual(len(attempts), 2)
            diagnostic = attempts[0].record["diagnostic"]
            self.assertEqual(diagnostic["evidence_quote"], "An invented deadline.")
            self.assertEqual(diagnostic["evidence_chunk_ids"], ["policy::0000"])
            self.assertEqual(diagnostic["purpose"], "reference_grounding")

    def test_legacy_score_row_is_not_counted_as_complete(self):
        self.dataset()
        run_id = freeze_run("questions-v1", "legacy", conditions=["C0"])
        run_experiment(run_id, pipeline=self.pipeline)
        with db.session_scope() as session:
            answer = session.query(Result).one()
            session.add(Score(answer_id=answer.id, metrics={"reference_groundedness": 1}, details={}))
        report = run_status(run_id)
        self.assertEqual(report["evaluation_records"], 1)
        self.assertEqual(report["automatically_scored"], 0)
        self.assertEqual(report["evaluation_legacy_or_invalid"], 1)

    def test_wrong_evaluation_protocol_is_rejected_without_model_calls(self):
        from ragapp.phase2.workflow import evaluate_run
        self.dataset()
        run_id = freeze_run("questions-v1", "old-protocol", conditions=["C0"])
        run_experiment(run_id, pipeline=self.pipeline)
        with db.session_scope() as session:
            run = session.get(Run, run_id)
            run.manifest = {**run.manifest, "evaluation_protocol": "old"}
        with self.assertRaisesRegex(ValueError, "staged evaluation protocol"):
            evaluate_run(run_id, evaluator=lambda *a, **k: self.fail("No model calls allowed"))

    def test_real_local_vector_ingest_and_freeze_consistency(self):
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from qdrant_client import QdrantClient
        from ragapp.phase2.provenance import ingest_corpus, verify_corpus as actual_verify
        from ragapp.shared.vectorstore import get_vectorstore
        corpus = self.root / "corpus"
        corpus.mkdir()
        document = corpus / "policy.md"
        document.write_text("---\ntitle: Test Policy\n---\n# Applications\nApplications close on Monday.", encoding="utf-8")
        client = QdrantClient(":memory:")
        get_vectorstore.cache_clear()
        try:
            with patch("ragapp.shared.vectorstore.get_qdrant_client", return_value=client), patch(
                "ragapp.shared.vectorstore.get_embeddings", return_value=DeterministicFakeEmbedding(size=1536)
            ):
                counts = ingest_corpus(corpus, True)
                self.assertEqual(counts["documents"], 1)
                chunks, fingerprint = actual_verify()
                self.assertEqual(chunks[0]["chunk_id"], "policy::0000")
                self.assertEqual(len(fingerprint), 64)
                with self.assertRaisesRegex(ValueError, "empty isolated"):
                    ingest_corpus(corpus, True)
                document.write_text("Changed document", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Source corpus files changed"):
                    actual_verify()
        finally:
            get_vectorstore.cache_clear()
            client.close()

    def test_end_to_end_graph_scoring_human_packet_and_analysis(self):
        import csv
        from ragapp.phase2.pipelines import run_pipeline
        from ragapp.phase2.evaluation import evaluate_answer
        from ragapp.phase2.manual import export_manual, import_manual
        from ragapp.phase2.workflow import evaluate_run, analyze_run
        from test_phase2_pipelines import ScriptedClient, ANSWER, SUFFICIENT, settings
        from test_phase2_evaluation import FixtureJudge
        # In-memory fixture, not a thesis result or production model experiment.
        self.chunks = [RetrievedChunk(chunk_id="a", document_title="Policy", text="Applications close Monday.").model_dump()]
        self.corpus.stop()
        self.corpus = patch("ragapp.phase2.runner.verify_corpus", return_value=(self.chunks, "corpus-1"))
        self.corpus.start()
        questions = [self.question(i, gold_chunk_ids=["a"]) for i in (1, 2, 3)]
        self.dataset(questions=questions)
        run_id = freeze_run("questions-v1", "e2e-offline")
        def pipeline(condition, question_id, text):
            replies = {"generate": [ANSWER]}
            if condition.value in {"C2", "C4"}:
                replies["rerank"] = [{"ranked_chunk_ids": ["a"]}]
            if condition.value in {"C3", "C4"}:
                replies["grade_evidence"] = [SUFFICIENT]
                replies["verify_draft"] = [{"fully_supported": True}]
            return run_pipeline(condition, question_id, text, client=ScriptedClient(replies),
                                dense=lambda *a, **k: [RetrievedChunk.model_validate(self.chunks[0])],
                                sparse=lambda *a, **k: [RetrievedChunk.model_validate(self.chunks[0])], settings=settings())
        run_experiment(run_id, pipeline=pipeline)
        report = evaluate_run(run_id, evaluator=lambda q, a, c, **kw: evaluate_answer(q, a, c, FixtureJudge(), **kw))
        self.assertTrue(report["evaluation_complete"])
        packet = self.root / "manual.csv"
        self.assertEqual(export_manual(run_id, packet)["rows"], 15)
        with packet.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["reference_groundedness"] = "1"
            row["action_appropriate"] = "1"
        with packet.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.assertEqual(import_manual(packet, "offline-fixture-rater", attest_human=True), 15)
        output = self.root / "analysis"
        result = analyze_run(run_id, output)
        self.assertEqual(result["human_agreement"], "available")
        self.assertEqual(result["n_questions"], 3)
        audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["independent_human_rows"], 15)
        self.assertEqual(audit["run"]["expected"], 15)
        self.assertTrue((output / "auto_manual_agreement.json").is_file())


class Phase2ConfigurationTests(unittest.TestCase):
    def test_configuration_does_not_load_legacy_env_or_keys(self):
        from ragapp.phase2.environment import configure
        with tempfile.TemporaryDirectory() as folder:
            envfile = Path(folder) / ".env.phase2"
            envfile.write_text("RAGAPP_DATABASE_URL=sqlite:///:memory:\nRAGAPP_QDRANT_URL=http://localhost:6335\nRAGAPP_LLM_MODEL=phase2-test-model\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                try:
                    settings = configure(envfile)
                    self.assertEqual(settings.llm_model, "phase2-test-model")
                    self.assertNotIn("OPENAI_API_KEY", os.environ)
                finally:
                    get_settings.cache_clear()

    def test_phase1_database_alias_rejected(self):
        from ragapp.phase2.environment import assert_isolated
        from ragapp.config import Settings
        settings = Settings(_env_file=None, database_url="postgresql+psycopg2://user:secret@127.0.0.1:5432/ragapp")
        with patch("ragapp.phase2.environment.dotenv_values", return_value={}):
            with self.assertRaisesRegex(ValueError, "Phase 1 database"):
                assert_isolated(settings)

    def test_gpt54_sampling_is_explicitly_non_reasoning(self):
        from ragapp.shared.llm import chat_options
        self.assertEqual(chat_options("gpt-5.4-mini", 0.0)["reasoning_effort"], "none")
        self.assertEqual(chat_options("gpt-5.4", 0.0)["temperature"], 0.0)
        self.assertNotIn("temperature", chat_options("gpt-5.4", 0.0, "high"))
        self.assertNotIn("temperature", chat_options("gpt-5-mini", 0.0))

    def test_dataset_gold_validation(self):
        with self.assertRaises(ValueError):
            QuestionRecord(id=1, question="?", qtype="single_doc", is_answerable=True,
                           expected_action="ANSWER", gold_reviewed=True)


if __name__ == "__main__":
    unittest.main()
