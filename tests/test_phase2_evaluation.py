import unittest
import copy
import json
from collections import Counter

from ragapp.phase2.contracts import PipelineResult, QuestionRecord
from ragapp.phase2.evaluation import EvaluationJudge, evaluate_answer, quote_occurs_in
from ragapp.phase2.evaluation_state import evaluation_status
from ragapp.schemas import CitedSentence, RetrievedChunk


class FixtureJudge:
    def __init__(self, labels=None, clarify=True):
        self.labels = labels or ["supported"]
        self.clarify = clarify
        self.purposes = []

    def decompose(self, answer):
        return [{"text": sentence.text, "sentence_index": index} for index, sentence in enumerate(answer.sentences)]

    def verify(self, requests, passages, purpose, repair=False):
        self.purposes.append(purpose)
        by_id = {p.chunk_id: p for p in passages}
        rows = []
        for r in requests:
            ids = r["allowed_chunk_ids"]
            label = self.labels[r["claim_index"] % len(self.labels)] if ids else "unsupported"
            rows.append({"claim_index": r["claim_index"], "verdict": label,
                         "evidence_chunk_ids": ids[:1] if label != "unsupported" else [],
                         "evidence_quotes": [by_id[ids[0]].text] if label != "unsupported" else [],
                         "rationale": "Offline test fixture"})
        return rows

    def citation_links(self, links):
        return [{"link_index": link["link_index"], "supports": True, "evidence_quote": link["passage"],
                 "rationale": "Offline fixture"} for link in links]

    def behaviour(self, question, answer):
        return {"action_appropriate": answer.action in question.acceptable_actions,
                "clarification_resolves_missing_slot": self.clarify, "rationale": "Offline fixture"}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.doc = RetrievedChunk(chunk_id="a", document_title="Policy", text="Monday is the deadline.")
        self.question = QuestionRecord(id=1, question="When is the deadline?", qtype="single_doc", is_answerable=True,
            expected_action="ANSWER", gold_answer="Monday", gold_chunk_ids=["a"], required_facts=["Monday"],
            gold_reviewed=True, family_id="deadline", annotation_notes="Human checked.")

    def answer(self, condition="C1", sentences=None, **updates):
        return PipelineResult(condition=condition, question_id=1, action=updates.pop("action", "ANSWER"),
            answer_text=updates.pop("answer_text", "Monday is the deadline."),
            sentences=sentences if sentences is not None else [CitedSentence(text="Monday is the deadline.", citations=["a"])],
            retrieved_chunks=[self.doc] if condition != "C0" else [], **updates)

    def test_c0_and_rag_share_reference_standard(self):
        c0 = evaluate_answer(self.question, self.answer("C0"), [self.doc], FixtureJudge())["metrics"]
        c1 = evaluate_answer(self.question, self.answer(), [self.doc], FixtureJudge())["metrics"]
        self.assertEqual(c0["reference_groundedness"], c1["reference_groundedness"])
        self.assertIsNone(c0["citation_precision"])
        self.assertIsNone(c0["retrieved_context_faithfulness"])

    def test_partial_is_not_silently_unsupported(self):
        sentences = [CitedSentence(text=x, citations=["a"]) for x in ("claim1", "claim2", "claim3", "claim4")]
        scores = evaluate_answer(self.question, self.answer(sentences=sentences), [self.doc],
            FixtureJudge(["supported", "partially_supported", "unsupported", "contradicted"]))["metrics"]
        self.assertEqual(scores["reference_groundedness"], .25)
        self.assertEqual(scores["reference_weighted_support"], .375)
        self.assertEqual(scores["hallucination_rate"], .5)
        self.assertEqual(scores["partially_supported_claims"], 1)
        self.assertEqual(scores["unsupported_claims"], 1)

    def test_refusal_without_claims_has_null_quality(self):
        result = evaluate_answer(self.question, self.answer(action="REFUSE", answer_text="", sentences=[], reason="No evidence."),
                                 [self.doc], FixtureJudge())["metrics"]
        for name in ("reference_groundedness", "hallucination_rate", "citation_completeness"):
            self.assertIsNone(result[name])
        self.assertTrue(result["false_refusal"])
        self.assertFalse(result["answerable_coverage"])

    def test_missing_citations_yield_zero_completeness_not_precision_one(self):
        answer = self.answer(sentences=[CitedSentence(text="Monday.", citations=[])])
        scores = evaluate_answer(self.question, answer, [self.doc], FixtureJudge())["metrics"]
        self.assertIsNone(scores["citation_precision"])
        self.assertEqual(scores["citation_completeness"], 0.0)

    def test_fabricated_ids_penalize_precision(self):
        answer = self.answer(sentences=[CitedSentence(text="Monday.", citations=["a", "invented"])])
        scores = evaluate_answer(self.question, answer, [self.doc], FixtureJudge())["metrics"]
        self.assertEqual(scores["citation_precision"], .5)
        self.assertEqual(scores["citation_validity"], .5)

    def test_alternative_evidence_is_not_mistaken_for_missing_retrieval(self):
        alternative = self.doc.model_copy(update={"chunk_id": "b"})
        question = self.question.model_copy(update={"gold_chunk_ids": ["a", "b"], "gold_evidence_groups": [["a", "b"]]})
        scores = evaluate_answer(question, self.answer(), [self.doc, alternative], FixtureJudge())["metrics"]
        self.assertEqual(scores["retrieval_recall"], 1.0)
        self.assertEqual(scores["gold_citation_overlap_recall"], .5)

    def test_generic_clarification_does_not_get_credit(self):
        question = QuestionRecord.model_validate({**self.question.model_dump(mode="json"),
            "qtype": "ambiguous", "expected_action": "CLARIFY",
            "acceptable_actions": ["CLARIFY"], "missing_slots": ["programme"]})
        answer = self.answer(action="CLARIFY", answer_text="", sentences=[], clarification_question="Can you say more?")
        scores = evaluate_answer(question, answer, [self.doc], FixtureJudge(clarify=False))["metrics"]
        self.assertFalse(scores["appropriate_clarification"])
        self.assertFalse(scores["action_appropriate"])

    def test_missing_reference_is_not_scored_as_hallucination(self):
        with self.assertRaisesRegex(ValueError, "reference evidence"):
            evaluate_answer(self.question, self.answer(), [], FixtureJudge())

    def test_judge_missing_claims_fails_closed(self):
        judge = FixtureJudge()
        judge.verify = lambda *args: []
        result = evaluate_answer(self.question, self.answer(), [self.doc], judge)
        self.assertIsNone(result["metrics"]["reference_groundedness"])
        self.assertTrue(result["metrics"]["action_appropriate"])
        self.assertIn("omitted", result["details"]["evaluation"]["stages"]["reference_grounding"]["reason"])

    def test_judge_invented_quote_fails_closed(self):
        judge = FixtureJudge()
        original = judge.verify
        def bad(*args):
            rows = original(*args)
            rows[0]["evidence_quotes"] = ["Invented quotation."]
            return rows
        judge.verify = bad
        result = evaluate_answer(self.question, self.answer(), [self.doc], judge)
        self.assertIsNone(result["metrics"]["hallucination_rate"])
        self.assertIsNone(result["metrics"]["unsupported_claims"])
        self.assertTrue(result["metrics"]["action_appropriate"])
        stage = result["details"]["evaluation"]["stages"]["reference_grounding"]
        self.assertIn("quote does not occur", stage["reason"])
        self.assertEqual(stage["attempts"][0]["diagnostic"]["evidence_quote"], "Invented quotation.")

    def test_quote_match_tolerates_wrapping_and_whitespace(self):
        passage = "Monday is the deadline."
        self.assertTrue(quote_occurs_in('  “Monday  is the deadline.” ', [passage]))
        self.assertTrue(quote_occurs_in("Monday is the deadline", [passage]))
        self.assertTrue(quote_occurs_in("Wi‑Fi capability", ["Wi-Fi capability, and mouse"]))
        self.assertFalse(quote_occurs_in("Tuesday is the deadline.", [passage]))
        self.assertFalse(quote_occurs_in("Mon", [passage]))

    def test_whitespace_normalized_judge_quote_is_accepted(self):
        judge = FixtureJudge()
        original = judge.verify
        def messy(*args):
            rows = original(*args)
            rows[0]["evidence_quotes"] = ['“Monday   is the deadline.”']
            return rows
        judge.verify = messy
        scores = evaluate_answer(self.question, self.answer(), [self.doc], judge)["metrics"]
        self.assertEqual(scores["reference_groundedness"], 1.0)

    def test_short_quotes_and_numeric_punctuation_are_not_conflated(self):
        self.assertTrue(quote_occurs_in("8GB", ["memory 8GB (preferably 16GB)"]))
        self.assertTrue(quote_occurs_in("Wi-Fi", ["Wi-Fi capability"]))
        self.assertFalse(quote_occurs_in("8GB", ["memory 18GB"]))
        self.assertFalse(quote_occurs_in("A fee of 1,000 dollars", ["A fee of 1.000 dollars"]))
        self.assertFalse(quote_occurs_in("memory 8GB...Wi-Fi", ["memory 8GB and a camera with Wi-Fi"]))
        self.assertFalse(quote_occurs_in("", ["anything"]))

    def test_only_failed_stage_retries_and_claims_are_fixed(self):
        judge = TrackingJudge({"reference_grounding": 1})
        result = evaluate_answer(self.question, self.answer(), [self.doc], judge, max_attempts=2)
        self.assertEqual(evaluation_status(**result), "complete")
        self.assertEqual(judge.counts["claims"], 1)
        self.assertEqual(judge.counts["reference_grounding"], 2)
        self.assertEqual(judge.counts["retrieved_context_support"], 1)
        self.assertEqual(judge.counts["behaviour"], 1)
        self.assertEqual(len(result["details"]["evaluation"]["stages"]["reference_grounding"]["attempts"]), 2)

    def test_partial_stage_resume_does_not_repeat_successes(self):
        first_judge = TrackingJudge({"citation_links": 10})
        first = evaluate_answer(self.question, self.answer(), [self.doc], first_judge, max_attempts=2)
        self.assertEqual(evaluation_status(**first), "partial")
        self.assertEqual(first["metrics"]["reference_groundedness"], 1)
        self.assertIsNone(first["metrics"]["citation_precision"])
        self.assertEqual(first["details"]["evaluation"]["metric_states"]["citation_precision"]["status"], "evaluation_failed")
        quiet = TrackingJudge()
        unchanged = evaluate_answer(self.question, self.answer(), [self.doc], quiet, previous=first, max_attempts=2)
        self.assertFalse(quiet.counts)
        second = evaluate_answer(self.question, self.answer(), [self.doc], quiet, previous=unchanged,
                                 max_attempts=2, retry_errors=True)
        self.assertEqual(evaluation_status(**second), "complete")
        self.assertEqual(quiet.counts, {"citation_links": 1})
        self.assertEqual(first["details"]["claims"], second["details"]["claims"])
        self.assertEqual(len(second["details"]["evaluation"]["stages"]["citation_links"]["attempts"]), 3)

    def test_claim_failure_blocks_dependents_but_not_behaviour_or_metadata(self):
        judge = TrackingJudge({"claims": 10})
        result = evaluate_answer(self.question, self.answer(), [self.doc], judge, max_attempts=2)
        self.assertTrue(result["metrics"]["action_appropriate"])
        self.assertEqual(result["metrics"]["retrieval_recall"], 1)
        self.assertIsNone(result["metrics"]["hallucination_rate"])
        states = result["details"]["evaluation"]["metric_states"]
        self.assertEqual(states["total_claims"]["status"], "evaluation_failed")
        self.assertEqual(states["hallucination_rate"]["status"], "blocked")
        self.assertEqual(judge.counts, {"claims": 2, "behaviour": 1})

    def test_resume_rejects_changed_answer(self):
        first = evaluate_answer(self.question, self.answer(), [self.doc], TrackingJudge())
        with self.assertRaisesRegex(ValueError, "input changed"):
            evaluate_answer(self.question, self.answer(answer_text="Different answer"), [self.doc], TrackingJudge(), previous=first)

    def test_checkpoint_interruption_preserves_claims_and_finished_reference(self):
        saved = []
        def checkpoint(result):
            saved.append(result)
            if result["details"]["evaluation"]["stages"]["reference_grounding"]["status"] == "success":
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            evaluate_answer(self.question, self.answer(), [self.doc], TrackingJudge(), on_checkpoint=checkpoint)
        judge = TrackingJudge()
        resumed = evaluate_answer(self.question, self.answer(), [self.doc], judge, previous=saved[-1])
        self.assertEqual(evaluation_status(**resumed), "complete")
        self.assertEqual(judge.counts["claims"], 0)
        self.assertEqual(judge.counts["reference_grounding"], 0)

    def test_interrupted_attempt_consumes_budget_and_dependencies_wait(self):
        saved = []
        def checkpoint(result):
            saved.append(result)
            if result["details"]["evaluation"]["stages"]["claims"]["status"] == "running":
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            evaluate_answer(self.question, self.answer(), [self.doc], TrackingJudge(), on_checkpoint=checkpoint)
        judge = TrackingJudge()
        resumed = evaluate_answer(self.question, self.answer(), [self.doc], judge, previous=saved[-1])
        self.assertEqual(judge.counts["claims"], 0)
        stage = resumed["details"]["evaluation"]["stages"]["claims"]
        self.assertEqual(stage["attempts"][0]["error_type"], "InterruptedEvaluation")
        self.assertEqual(stage["status"], "failed")
        repaired = evaluate_answer(self.question, self.answer(), [self.doc], judge, previous=resumed, retry_errors=True)
        self.assertEqual(evaluation_status(**repaired), "complete")

    def test_persistence_errors_are_not_swallowed_as_model_errors(self):
        def unavailable(_):
            raise OSError("Database unavailable")
        with self.assertRaisesRegex(OSError, "Database unavailable"):
            evaluate_answer(self.question, self.answer(), [self.doc], TrackingJudge(), on_checkpoint=unavailable)

    def test_api_error_body_is_not_persisted(self):
        judge = TrackingJudge({"behaviour": 5})
        result = evaluate_answer(self.question, self.answer(), [self.doc], judge)
        self.assertNotIn("private-token", str(result))
        self.assertEqual(result["metrics"]["reference_groundedness"], 1)
        self.assertIsNone(result["metrics"]["action_appropriate"])

    def test_empty_claims_are_not_applicable_and_not_evaluation_failure(self):
        answer = self.answer(action="REFUSE", answer_text="", sentences=[], reason="No evidence.")
        result = evaluate_answer(self.question, answer, [self.doc], TrackingJudge())
        self.assertEqual(evaluation_status(**result), "complete")
        state = result["details"]["evaluation"]["metric_states"]["hallucination_rate"]
        self.assertEqual(state["status"], "not_applicable")
        self.assertEqual(state["reason"], "no_factual_claims")

    def test_complete_flag_does_not_override_incomplete_metrics(self):
        result = evaluate_answer(self.question, self.answer(), [self.doc], TrackingJudge())
        broken = copy.deepcopy(result)
        broken["metrics"]["reference_groundedness"] = None
        self.assertEqual(evaluation_status(**broken), "invalid")

    def test_real_judge_adapter_receives_quote_repair_feedback_and_records_usage(self):
        class Client:
            def __init__(self, passage):
                self.passage, self.calls, self.inputs = passage, [], []

            def invoke(self, schema, messages, purpose):
                data = json.loads(messages[-1][1])
                self.inputs.append((purpose, data))
                self.calls.append({"purpose": purpose, "input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
                if purpose == "extract_claims":
                    return {"claims": [{"text": "Monday is the deadline.", "sentence_index": 0}]}
                if purpose == "action_rubric":
                    return {"action_appropriate": True, "clarification_resolves_missing_slot": False, "rationale": "fixture"}
                first_reference = purpose == "reference_grounding" and sum(c["purpose"] == purpose for c in self.calls) == 1
                return {"verdicts": [{"claim_index": 0, "verdict": "supported", "evidence_chunk_ids": ["a"],
                    "evidence_quotes": ["Invented quotation." if first_reference else self.passage], "rationale": "fixture"}]}

            def summary(self):
                return {"calls": self.calls}

        client = Client(self.doc.text)
        result = evaluate_answer(self.question, self.answer("C0"), [self.doc], EvaluationJudge(client), max_attempts=2)
        self.assertEqual(evaluation_status(**result), "complete")
        requests = [data for purpose, data in client.inputs if purpose == "reference_grounding"]
        self.assertNotIn("validation_feedback", requests[0])
        self.assertEqual(requests[1]["validation_feedback"]["diagnostic"]["evidence_quote"], "Invented quotation.")
        action_request = next(data for purpose, data in client.inputs if purpose == "action_rubric")
        self.assertNotIn("validation_feedback", action_request)
        self.assertEqual(result["details"]["judge_usage"]["llm_calls"], 4)
        self.assertEqual(result["details"]["judge_usage"]["total_tokens"], 60)

    def test_all_judge_stages_failing_still_retains_response_and_evidence_record(self):
        judge = TrackingJudge({"claims": 5, "behaviour": 5})
        result = evaluate_answer(self.question, self.answer(), [self.doc], judge)
        self.assertEqual(evaluation_status(**result), "failed")
        self.assertTrue(result["metrics"]["answered"])
        self.assertEqual(result["metrics"]["retrieval_recall"], 1)
        self.assertIsNone(result["metrics"]["action_appropriate"])


class TrackingJudge(FixtureJudge):
    def __init__(self, fail=None):
        super().__init__()
        self.fail = dict(fail or {})
        self.counts = Counter()

    def touch(self, name):
        self.counts[name] += 1
        if self.fail.get(name, 0) > 0:
            self.fail[name] -= 1
            raise TimeoutError("private-token must not be persisted")

    def decompose(self, answer):
        self.touch("claims")
        return super().decompose(answer)

    def verify(self, requests, passages, purpose):
        self.touch(purpose)
        return super().verify(requests, passages, purpose)

    def citation_links(self, links):
        self.touch("citation_links")
        return super().citation_links(links)

    def behaviour(self, question, answer):
        self.touch("behaviour")
        return super().behaviour(question, answer)


if __name__ == "__main__":
    unittest.main()
