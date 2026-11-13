import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ragapp.phase2.analysis import (agreement_pair, analyze_scores, auto_manual_agreement,
    descriptives, holm_adjust, paired_comparison, planned_contrasts)


def frame():
    return pd.DataFrame([dict(question_id=q, condition=c, qtype="single_doc", expected_action="ANSWER",
        split="test", status="success", action="ANSWER", answer_id=f"{q}-{c}",
        reference_groundedness=0.5 if c == "C1" else 0.75, answerable_coverage=1, hallucinated=False)
        for q in (1, 2, 3) for c in ("C0", "C1", "C2", "C3", "C4")])


class AnalysisTests(unittest.TestCase):
    def test_exact_mcnemar_uses_discordant_pairs(self):
        result = paired_comparison([1, 1, 1, 1], [0, 0, 0, 0], binary=True, bootstrap_samples=100)
        self.assertEqual(result["a_only"], 4)
        self.assertEqual(result["p"], .125)
        self.assertEqual(result["mean_difference"], 1.0)

    def test_zero_differences_are_valid(self):
        result = paired_comparison([.4, .7], [.4, .7], bootstrap_samples=100)
        self.assertEqual(result["p"], 1.0)
        self.assertEqual(result["mean_difference"], 0.0)

    def test_paired_bootstrap_reproducible(self):
        self.assertEqual(paired_comparison([1, 2, 4], [0, 2, 2], seed=9, bootstrap_samples=100),
                         paired_comparison([1, 2, 4], [0, 2, 2], seed=9, bootstrap_samples=100))

    def test_holm_includes_prespecified_unavailable_tests(self):
        self.assertEqual(holm_adjust([.01, .04, None]), [.03, .08, None])

    def test_pairwise_not_global_complete_case(self):
        df = frame()
        df.loc[df.condition == "C4", "reference_groundedness"] = None
        row = next(r for r in planned_contrasts(df, bootstrap_samples=100)
                   if r["condition_a"] == "C1" and r["condition_b"] == "C0")
        self.assertEqual(row["n_pairs"], 3)

    def test_kappa_constant_labels_is_undefined_not_perfect(self):
        result = agreement_pair([1, 1], [1, 1], binary=True)
        self.assertIsNone(result["cohen_kappa"])
        self.assertEqual(result["raw_agreement"], 1.0)

    def test_no_human_data_does_not_invent_agreement(self):
        self.assertEqual(auto_manual_agreement(frame(), pd.DataFrame())["status"], "not_available")

    def test_ai_assisted_ratings_are_excluded(self):
        manual = pd.DataFrame([dict(answer_id="1-C1", evaluator="x", source="ai_assisted", reference_groundedness=.5)])
        self.assertEqual(auto_manual_agreement(frame(), manual)["status"], "not_available")

    def test_human_pairing_and_bias(self):
        manual = pd.DataFrame([dict(answer_id=f"{q}-C1", evaluator="rater", source="human", reference_groundedness=.25)
                               for q in (1, 2, 3)])
        result = auto_manual_agreement(frame(), manual)
        overall = next(r for r in result["comparisons"] if r["dimension"] == "overall")
        self.assertEqual(overall["n"], 3)
        self.assertEqual(overall["bias_auto_minus_human"], .25)

    def test_human_comments_without_scores_do_not_establish_agreement(self):
        manual = pd.DataFrame([dict(answer_id="1-C1", evaluator="human", source="human", notes="Reviewed later.",
                                    reference_groundedness=None)])
        self.assertEqual(auto_manual_agreement(frame(), manual)["status"], "not_available")

    def test_json_exports_are_strict_and_sample_is_question_count(self):
        with tempfile.TemporaryDirectory() as folder:
            result = analyze_scores(frame(), Path(folder), bootstrap_samples=100)
            self.assertEqual(result["n_questions"], 3)
            data = (Path(folder) / "tests.json").read_text()
            self.assertNotIn("NaN", data)
            self.assertEqual(json.loads(data)["n_result_cells"], 15)
            self.assertTrue((Path(folder) / "coverage_and_actions.csv").exists())

    def test_mixed_splits_rejected(self):
        df = frame()
        df.loc[0, "split"] = "development"
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "pool"):
                analyze_scores(df, folder, bootstrap_samples=100)

    def test_required_fact_coverage_is_descriptive_by_human_rater(self):
        manual = pd.DataFrame([
            dict(answer_id="1-C1", evaluator="rater-a", source="human", required_fact_coverage=.5),
            dict(answer_id="1-C1", evaluator="rater-b", source="human", required_fact_coverage=1.0),
            dict(answer_id="2-C1", evaluator="assisted", source="ai_assisted", required_fact_coverage=0),
        ])
        with tempfile.TemporaryDirectory() as folder:
            analyze_scores(frame(), Path(folder), manual_df=manual, bootstrap_samples=100)
            detail = pd.read_csv(Path(folder) / "manual_required_fact_coverage.csv")
            self.assertEqual(set(detail.evaluator), {"rater-a", "rater-b"})
            self.assertEqual(len(detail), 2)
            summary = pd.read_csv(Path(folder) / "manual_required_fact_coverage_summary.csv")
            self.assertEqual(len(summary), 2)
            tests = json.loads((Path(folder) / "tests.json").read_text())
            self.assertFalse(any(r["metric"] == "required_fact_coverage" for r in tests["contrasts"]))

    def test_missingness_distinguishes_failure_from_inapplicability(self):
        df = frame()
        df["reference_groundedness__status"] = "ok"
        for q, state in ((1, "evaluation_failed"), (2, "not_applicable")):
            cells = (df.question_id == q) & (df.condition == "C4")
            df.loc[cells, "reference_groundedness"] = None
            df.loc[cells, "reference_groundedness__status"] = state
        row = next(r for r in planned_contrasts(df, bootstrap_samples=100)
                   if r["metric"] == "reference_groundedness" and r["condition_a"] == "C4" and r["condition_b"] == "C2")
        self.assertEqual(row["n_pairs"], 1)
        self.assertEqual(row["n_excluded_questions_with_evaluation_failure"], 1)
        self.assertEqual(row["n_excluded_questions_with_not_applicable"], 1)
        rows = descriptives(df, ["condition"])
        desc = rows[(rows.condition == "C4") & (rows.metric == "reference_groundedness")].iloc[0]
        self.assertEqual(desc.n_evaluation_failed_or_blocked, 1)
        self.assertEqual(desc.n_not_applicable, 1)
        self.assertEqual(desc.n_observed, 1)

    def test_invalid_non_null_metric_cannot_enter_analysis(self):
        df = frame()
        df["reference_groundedness__status"] = "ok"
        df.loc[df.condition == "C4", "reference_groundedness__status"] = "evaluation_failed"
        with tempfile.TemporaryDirectory() as folder:
            analyze_scores(df, folder, bootstrap_samples=100)
            data = json.loads((Path(folder) / "tests.json").read_text())
            row = next(r for r in data["contrasts"] if r["metric"] == "reference_groundedness"
                       and r["condition_a"] == "C4" and r["condition_b"] == "C2")
            self.assertEqual(row["n_pairs"], 0)
            availability = pd.read_csv(Path(folder) / "metric_availability.csv")
            failed = availability[(availability.condition == "C4") & (availability.metric == "reference_groundedness")]
            self.assertTrue(failed.value.isna().all())


if __name__ == "__main__":
    unittest.main()
