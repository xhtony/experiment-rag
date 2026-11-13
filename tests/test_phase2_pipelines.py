"""Pipeline behavior tested through real LangGraph execution with local fixtures."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ragapp.phase2.pipelines import Decision, bounded_context, run_pipeline
from ragapp.schemas import RetrievedChunk


def settings(**updates):
    values = dict(top_k=2, candidate_k=4, rrf_k=60, max_context_chars=12000, c3_max_retries=1)
    values.update(updates)
    return SimpleNamespace(**values)


def chunk(cid="a", text="Applications close on Monday."):
    return RetrievedChunk(chunk_id=cid, document_title="Test", text=text)


class ScriptedClient:
    def __init__(self, responses):
        self.responses = {key: list(value) for key, value in responses.items()}
        self.calls = []

    def invoke(self, schema, messages, purpose):
        self.calls.append((purpose, messages))
        if purpose not in self.responses or not self.responses[purpose]:
            raise AssertionError(f"Unexpected call: {purpose}")
        response = self.responses[purpose].pop(0)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)

    def summary(self):
        return {"llm_calls": len(self.calls), "input_tokens": None, "output_tokens": None}


ANSWER = {"action": "ANSWER", "sentences": [{"text": "Applications close on Monday.", "citations": ["a"]}]}
SUFFICIENT = {"sufficient": True, "needs_clarification": False}
INSUFFICIENT = {"sufficient": False, "needs_clarification": False, "missing_info": "deadline"}


class PipelineTests(unittest.TestCase):
    def test_c0_does_not_retrieve_and_can_clarify(self):
        client = ScriptedClient({"generate": [{"action": "CLARIFY", "clarification_question": "Which programme?"}]})
        dense, sparse = Mock(side_effect=AssertionError("no retrieval")), Mock(side_effect=AssertionError("no retrieval"))
        result = run_pipeline("C0", 1, "What is my deadline?", client=client, dense=dense, sparse=sparse, settings=settings())
        self.assertEqual(result.action.value, "CLARIFY")
        dense.assert_not_called()
        sparse.assert_not_called()
        self.assertEqual(result.retrieved_chunks, [])

    def test_c1_allows_refusal(self):
        client = ScriptedClient({"generate": [{"action": "REFUSE", "reason": "Insufficient evidence."}]})
        result = run_pipeline("C1", 1, "Unknown policy?", client=client, dense=lambda *a, **k: [], settings=settings())
        self.assertEqual(result.action.value, "REFUSE")
        self.assertEqual(result.usage["llm_calls"], 1)

    def test_c2_reranking_deduplicates_and_records_unknown_ids(self):
        client = ScriptedClient({"rerank": [{"ranked_chunk_ids": ["b", "b", "invented"]}], "generate": [ANSWER]})
        result = run_pipeline("C2", 1, "Deadline?", client=client, dense=lambda *a, **k: [chunk("a")],
                              sparse=lambda *a, **k: [chunk("b")], settings=settings())
        self.assertEqual([c.chunk_id for c in result.retrieved_chunks], ["b", "a"])
        self.assertEqual(result.trace["retrieval_attempts"][0]["reranker_unknown_ids"], ["invented"])

    def test_c4_corrective_retry_stays_hybrid(self):
        client = ScriptedClient({"rerank": [{"ranked_chunk_ids": ["a"]}, {"ranked_chunk_ids": ["a"]}],
            "grade_evidence": [INSUFFICIENT, SUFFICIENT], "rewrite_query": [{"query": "application deadline"}],
            "generate": [ANSWER], "verify_draft": [{"fully_supported": True}]})
        dense = Mock(return_value=[chunk()])
        sparse = Mock(return_value=[chunk()])
        result = run_pipeline("C4", 1, "When can I apply?", client=client, dense=dense, sparse=sparse, settings=settings())
        self.assertEqual(dense.call_count, 2)
        self.assertEqual(sparse.call_count, 2)
        self.assertEqual([t["strategy"] for t in result.trace["retrieval_attempts"]], ["hybrid", "hybrid"])
        self.assertEqual(result.trace["retrieval_attempts"][1]["query"], "application deadline")
        self.assertIn("draft_before_verification", result.trace)

    def test_c3_and_c4_share_clarification_rule(self):
        for condition in ("C3", "C4"):
            replies = {"grade_evidence": [{"sufficient": False, "needs_clarification": True,
                        "clarification_question": "Which programme and intake?"}]}
            if condition == "C4":
                replies["rerank"] = [{"ranked_chunk_ids": ["a"]}]
            client = ScriptedClient(replies)
            result = run_pipeline(condition, 1, "My deadline?", client=client, dense=lambda *a, **k: [chunk()],
                                  sparse=lambda *a, **k: [chunk()], settings=settings())
            self.assertEqual(result.action.value, "CLARIFY")
            self.assertEqual(result.clarification_question, "Which programme and intake?")
            self.assertNotIn("rewrite_query", [purpose for purpose, _ in client.calls])

    def test_failed_verification_preserves_rejected_draft(self):
        client = ScriptedClient({"grade_evidence": [SUFFICIENT], "generate": [ANSWER],
                                 "verify_draft": [{"fully_supported": False, "problem": "Unstated date."}]})
        result = run_pipeline("C3", 1, "When?", client=client, dense=lambda *a, **k: [chunk()], settings=settings())
        self.assertEqual(result.action.value, "REFUSE")
        self.assertEqual(result.sentences, [])
        self.assertEqual(result.trace["draft_before_verification"]["sentences"][0]["text"], ANSWER["sentences"][0]["text"])

    def test_exhausted_evidence_does_not_generate_an_answer(self):
        client = ScriptedClient({"grade_evidence": [INSUFFICIENT, INSUFFICIENT], "rewrite_query": [{"query": "policy"}]})
        result = run_pipeline("C3", 1, "Question", client=client, dense=lambda *a, **k: [], settings=settings())
        self.assertEqual(result.action.value, "REFUSE")
        self.assertEqual(len(result.trace["retrieval_attempts"]), 2)

    def test_technical_failure_is_propagated_not_refused(self):
        client = ScriptedClient({"generate": [TimeoutError("offline fixture")]})
        with self.assertRaises(TimeoutError):
            run_pipeline("C0", 1, "Question", client=client, settings=settings())

    def test_context_budget_counts_headers_and_separators(self):
        from ragapp.shared.prompts import format_context
        docs = bounded_context([chunk("a", "a"*1000), chunk("a"), chunk("b")], 3, 170)
        self.assertLessEqual(len(format_context(docs)), 170)
        self.assertEqual(len({c.chunk_id for c in docs}), len(docs))

    def test_schema_rejects_inconsistent_action(self):
        with self.assertRaises(ValueError):
            Decision(action="ANSWER", sentences=[])
        with self.assertRaises(ValueError):
            Decision(action="CLARIFY", clarification_question="")
        with self.assertRaises(ValueError):
            Decision(action="REFUSE", reason="No evidence", sentences=ANSWER["sentences"])


if __name__ == "__main__":
    unittest.main()
