"""Offline regression cases for v3 measurement contracts, not model accuracy tests."""
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from ragapp.phase2.contracts import PipelineResult, QuestionRecord
from ragapp.phase2.datasets import read_questions, validate_questions
from ragapp.phase2.evaluation import EvaluationJudge, evaluate_answer
from ragapp.phase2.evaluation_state import EVALUATION_PROTOCOL, evaluation_status
from ragapp.phase2 import prompts
from ragapp.schemas import CitedSentence, RetrievedChunk
from test_phase2_evaluation import FixtureJudge


class ReferenceProtocolTests(unittest.TestCase):
    def setUp(self):
        self.doc = RetrievedChunk(chunk_id="background", document_title="Submission policy",
                                  text="Submissions use a similarity checking service.")
        self.question = QuestionRecord(id=1, question="What exact score guarantees automatic acceptance?",
            qtype="unanswerable", is_answerable=False, expected_action="REFUSE",
            gold_answer="The supplied corpus does not establish that guarantee.",
            background_chunk_ids=["background"], family_id="acceptance-threshold",
            annotation_notes="Reviewed against fixture evidence.", gold_reviewed=True)

    def test_background_does_not_become_retrieval_gold(self):
        answer = PipelineResult(question_id=1, condition="C1", action="REFUSE",
            reason="Submissions use a similarity checking service, but no guarantee can be established.",
            retrieved_chunks=[self.doc])
        judge = FixtureJudge()
        judge.decompose = lambda answer: [{"text": self.doc.text, "sentence_index": -1}]
        result = evaluate_answer(self.question, answer, [self.doc], judge)
        self.assertEqual(evaluation_status(**result), "complete")
        self.assertEqual(result["metrics"]["reference_groundedness"], 1)
        self.assertEqual(result["metrics"]["hallucination_rate"], 0)
        self.assertTrue(result["metrics"]["appropriate_refusal"])
        for name in ("retrieval_recall", "candidate_recall", "gold_citation_overlap_recall"):
            self.assertIsNone(result["metrics"][name])
        for name in ("citation_precision", "citation_completeness"):
            self.assertIsNone(result["metrics"][name])
            self.assertEqual(result["details"]["evaluation"]["metric_states"][name]["reason"], "response_action_not_ANSWER")

    def test_refusal_background_does_not_excuse_invented_target(self):
        answer = PipelineResult(question_id=1, condition="C0", action="REFUSE", reason="Service used; the guaranteed cutoff is 12%.")
        judge = FixtureJudge(["supported", "unsupported"])
        judge.decompose = lambda answer: [{"text": self.doc.text, "sentence_index": -1},
                                         {"text": "The guaranteed cutoff is 12%.", "sentence_index": -1}]
        result = evaluate_answer(self.question, answer, [self.doc], judge)
        self.assertEqual(result["metrics"]["hallucination_rate"], .5)
        self.assertEqual(result["metrics"]["unsupported_claims"], 1)

    def test_background_missing_from_frozen_inputs_fails_before_judging(self):
        answer = PipelineResult(question_id=1, condition="C0", action="REFUSE")
        with self.assertRaisesRegex(ValueError, "reference evidence"):
            evaluate_answer(self.question, answer, [], FixtureJudge())

    def test_background_ids_are_validated_at_load(self):
        with self.assertRaisesRegex(ValueError, "Unknown reference chunk IDs"):
            validate_questions([self.question], set(), "test")
        validate_questions([self.question], {"background"}, "test")

    def test_background_annotations_are_disjoint_and_nonblank(self):
        data = self.question.model_dump(mode="json")
        for updates in ({"background_chunk_ids": ["background", "background"]},
                        {"gold_chunk_ids": ["background"]}, {"background_chunk_ids": [" "]}):
            with self.assertRaises(ValueError):
                QuestionRecord.model_validate({**data, **updates})

    def test_question_reaches_all_real_judge_adapters_without_gold_leak(self):
        class Client:
            def __init__(self):
                self.inputs = []
            def invoke(self, schema, messages, purpose):
                data = json.loads(messages[-1][1])
                self.inputs.append((purpose, data, messages[0][1]))
                if purpose == "extract_claims":
                    return {"claims": [{"text": "The charge is 60 dollars.", "sentence_index": 0}]}
                if purpose == "action_rubric":
                    return {"action_appropriate": True, "clarification_resolves_missing_slot": False, "rationale": "fixture"}
                if purpose == "citation_link_support":
                    return {"verdicts": [{"link_index": 0, "supports": True, "evidence_quote": "The charge is 10% of the amount.", "rationale": "fixture"}]}
                return {"verdicts": [{"claim_index": 0, "verdict": "supported", "evidence_chunk_ids": ["rule"],
                    "evidence_quotes": ["The charge is 10% of the amount."], "rationale": "10% of the hypothetical 600 is 60."}]}
            def summary(self):
                return {"calls": []}

        q = QuestionRecord(id=2, question="My amount is 600 dollars. What is the charge?", qtype="conditional",
            is_answerable=True, expected_action="ANSWER", gold_answer="60", required_facts=["Charge 60"],
            gold_chunk_ids=["rule"], family_id="amount-charge", annotation_notes="Reviewed fixture.", gold_reviewed=True)
        doc = RetrievedChunk(chunk_id="rule", document_title="Fees", text="The charge is 10% of the amount.")
        answer = PipelineResult(question_id=2, condition="C1", action="ANSWER", answer_text="The charge is 60 dollars.",
            sentences=[CitedSentence(text="The charge is 60 dollars.", citations=["rule"])], retrieved_chunks=[doc])
        client = Client()
        judge = EvaluationJudge(client, question_text="Stale question from another item")
        result = evaluate_answer(q, answer, [doc], judge)
        self.assertEqual(evaluation_status(**result), "complete")
        self.assertEqual(len(client.inputs), 6)
        for purpose, data, prompt in client.inputs:
            self.assertEqual(data["question"], q.question)
            self.assertNotIn("gold_answer", data)
            self.assertNotIn("required_facts", data)
            if purpose in {"reference_grounding", "retrieved_context_support", "citation_completeness", "citation_link_support"}:
                self.assertIn("hypothetical user circumstances", prompt)
                self.assertIn("derived", prompt)

    def test_old_protocol_checkpoint_is_not_reused(self):
        answer = PipelineResult(question_id=1, condition="C0", action="REFUSE")
        result = evaluate_answer(self.question, answer, [self.doc], FixtureJudge())
        result["details"]["evaluation"]["protocol"] = "phase2-evaluation-v2"
        with self.assertRaisesRegex(ValueError, "protocol"):
            evaluate_answer(self.question, answer, [self.doc], FixtureJudge(), previous=result)
        self.assertEqual(EVALUATION_PROTOCOL, "phase2-evaluation-v3")

    def test_same_clarification_policy_reaches_every_generation_prompt(self):
        for prompt in (prompts.C0, prompts.STANDARD, prompts.CORRECTIVE):
            self.assertIn("clarification-first", prompt)
            self.assertIn("conditional explanation", prompt)


class RevisedHeldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.folder = cls.root / "data/questions/phase2"
        cls.rows = read_questions(cls.folder / "heldout75.revised.draft.jsonl")
        cls.by_id = {q.id: q for q in cls.rows}

    def test_balanced_pending_review_candidate(self):
        self.assertEqual(len(self.rows), 75)
        self.assertEqual(set(self.by_id), set(range(201, 276)))
        self.assertEqual(set(Counter(q.qtype.value for q in self.rows).values()), {15})
        self.assertTrue(all(not q.gold_reviewed for q in self.rows))

    def test_unanswerable_background_is_separate(self):
        for q in self.rows:
            if q.qtype.value == "unanswerable":
                self.assertEqual(q.expected_action.value, "REFUSE")
                self.assertFalse(q.is_answerable)
                self.assertFalse(q.gold_chunk_ids)
                self.assertFalse(q.required_facts)
                self.assertTrue(q.background_chunk_ids)

    def test_actual_corpus_ids_and_reviewed_contracts(self):
        from ragapp.shared.corpus import load_corpus
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        ids = set()
        for doc in load_corpus(self.root / "data/corpus"):
            seq = 0
            for _, body, _ in doc.sections:
                for piece in splitter.split_text(body):
                    ids.add(f"{doc.slug}::{seq:04d}")
                    seq += 1
        validate_questions(self.rows, ids, "development")
        with self.assertRaisesRegex(ValueError, "human audit"):
            validate_questions(self.rows, ids, "test")
        reviewed = [QuestionRecord.model_validate({**q.model_dump(mode="json"), "gold_reviewed": True}) for q in self.rows]
        validate_questions(reviewed, ids, "test")

    def test_q227_requires_body_sequence_and_marking_guide(self):
        q = self.by_id[227]
        self.assertIn("comp901-thesis-assignment::0010", q.gold_chunk_ids)
        self.assertIn("comp901-thesis-marking-guide::0003", q.gold_chunk_ids)
        self.assertIn("immediately before", q.question)
        self.assertEqual(q.qtype.value, "multi_doc")

    def test_revision_and_human_review_hashes_match(self):
        data = (self.folder / "heldout75.revised.draft.jsonl").read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        history = json.loads((self.folder / "heldout75.revision-record.json").read_text(encoding="utf-8"))
        review = json.loads((self.folder / "heldout75.human-review.json").read_text(encoding="utf-8"))
        self.assertEqual(history["revised_sha256"], digest)
        self.assertEqual(review["dataset_sha256"], digest)
        self.assertEqual({r["id"] for r in review["questions"]}, set(self.by_id))
        self.assertFalse(history["human_approval"])

    def test_published_schema_matches_question_contract(self):
        schema = json.loads((self.folder / "question.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema, QuestionRecord.model_json_schema())


if __name__ == "__main__":
    unittest.main()
