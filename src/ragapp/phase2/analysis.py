"""Prespecified paired contrasts; missingness, coverage and agreement stay visible."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CONTINUOUS = ("reference_groundedness", "retrieved_context_faithfulness", "reference_weighted_support",
              "citation_precision", "citation_completeness", "citation_validity", "hallucination_rate",
              "retrieval_recall", "candidate_recall", "gold_citation_overlap_precision", "gold_citation_overlap_recall",
              "latency_ms", "input_tokens", "output_tokens", "total_tokens", "llm_calls")
BINARY = ("hallucinated", "action_appropriate", "answered", "answerable_coverage", "appropriate_refusal",
          "false_refusal", "appropriate_clarification", "unnecessary_clarification", "inappropriate_answer")


def analysis_plan():
    comparisons = []
    def add(family, metric, a, b, subset="all", interpretation="primary"):
        comparisons.append(dict(family=family, metric=metric, condition_a=a, condition_b=b,
                                subset=subset, role=interpretation, binary=metric in BINARY))
    for condition in ("C1", "C2", "C3"):
        add("RQ1", "reference_groundedness", condition, "C0")
    for metric in ("citation_precision", "citation_completeness"):
        add("RQ1", metric, "C2", "C1")
    add("RQ1", "hallucination_rate", "C3", "C1")
    add("RQ1", "appropriate_refusal", "C3", "C1", "unanswerable")
    add("RQ1", "appropriate_clarification", "C3", "C1", "ambiguous")
    add("RQ3", "retrieval_recall", "C4", "C3")
    add("RQ3", "citation_completeness", "C4", "C3")
    add("RQ3", "false_refusal", "C4", "C3", "answerable_multi_conditional")
    for subset in ("unanswerable", "ambiguous"):
        add("RQ3", "inappropriate_answer", "C4", "C2", subset)
    for baseline in ("C2", "C3"):
        for metric in ("reference_groundedness", "hallucination_rate", "citation_precision", "answerable_coverage"):
            add("RQ3", metric, "C4", baseline)
        for metric in ("latency_ms", "total_tokens", "llm_calls"):
            add("RQ3_operational", metric, "C4", baseline, interpretation="secondary")
    return {"version": "phase2-analysis-v3", "unit": "paired question", "contrasts": comparisons,
            "supplemental_manual_metric": "required_fact_coverage: fully met required facts / required facts; descriptive by rater, not an automated hypothesis endpoint",
            "alternatives": "two-sided", "alpha": 0.05, "correction": "Holm within each full prespecified family",
            "confidence_level": 0.95, "bootstrap_samples": 2000,
            "continuous_test": "Wilcoxon signed-rank, Pratt zeros; symmetry assumption applies to difference distribution",
            "binary_test": "exact McNemar using discordant paired observations",
            "effect": "mean paired difference a-b; binary difference is risk difference",
            "missingness": "pairwise observed applicable metrics; distinguish failed/blocked evaluation, pending, generation failure and genuine N/A; quality is conditional",
            "RQ2": "descriptive per-rater matched scores: Spearman, MAE, bias, raw agreement and Cohen kappa; no independent-row correlation p-values"}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def holm_adjust(pvalues):
    """Unestimable planned tests retain their place in the family as p=1."""
    values = [1.0 if p is None or not math.isfinite(p) else p for p in pvalues]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted, previous = [None] * len(values), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        if pvalues[index] is not None:
            adjusted[index] = previous
    return adjusted


def paired_comparison(a, b, binary=False, seed=42, bootstrap_samples=2000):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Paired comparison requires aligned finite observations")
    if binary and (not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all()):
        raise ValueError("Binary comparison received nonbinary values")
    n = len(a)
    if n == 0:
        return {"n_pairs": 0, "p": None, "mean_difference": None, "ci_low": None, "ci_high": None,
                "note": "No paired applicable observations"}
    diff = a - b
    result = {"n_pairs": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
              "mean_difference": float(diff.mean()), "ci_low": None, "ci_high": None, "p": None}
    if n >= 2:
        rng = np.random.default_rng(seed)
        estimates = diff[rng.integers(0, n, size=(bootstrap_samples, n))].mean(axis=1)
        result["ci_low"], result["ci_high"] = map(float, np.quantile(estimates, [0.025, 0.975]))
    if binary:
        a_only, b_only = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        discordant = a_only + b_only
        result.update(test="exact McNemar", a_only=a_only, b_only=b_only,
                      p=float(stats.binomtest(a_only, discordant, .5).pvalue) if discordant else 1.0)
    elif n >= 2:
        result["test"] = "Wilcoxon signed-rank"
        result["p"] = 1.0 if np.all(diff == 0) else float(stats.wilcoxon(a, b, zero_method="pratt", alternative="two-sided").pvalue)
        ranks = stats.rankdata(np.abs(diff[diff != 0]))
        result["paired_rank_biserial"] = float(np.sum(ranks * np.sign(diff[diff != 0])) / ranks.sum()) if len(ranks) else 0.0
    else:
        result["note"] = "Too few observations for the planned continuous test"
    return result


def _subset(df, name):
    if name == "all":
        return df
    if name == "answerable_multi_conditional":
        return df[df.qtype.isin(["multi_doc", "conditional"]) & (df.expected_action == "ANSWER")]
    return df[df.qtype == name]


def metric_status(df, metric):
    """Fallback labels serve standalone/legacy analysis frames, never certify missing scores."""
    values = pd.to_numeric(df[metric], errors="coerce") if metric in df else pd.Series(np.nan, index=df.index)
    result = pd.Series(np.where(np.isfinite(values), "ok", "unknown_missing"), index=df.index)
    if metric + "__status" in df:
        result = df[metric + "__status"].fillna("unknown_missing")
    return result


def missing_counts(statuses):
    return {"n_evaluation_failed_or_blocked": int(statuses.isin(["evaluation_failed", "blocked"]).sum()),
            "n_not_applicable": int((statuses == "not_applicable").sum()),
            "n_pending": int((statuses == "pending").sum()),
            "n_generation_failed": int((statuses == "generation_failed").sum()),
            "n_other_missing": int((~statuses.isin(["ok", "not_applicable", "evaluation_failed", "blocked", "pending", "generation_failed"])).sum())}


def metric_availability(df):
    rows = []
    for metric in (*CONTINUOUS, *BINARY):
        if metric not in df:
            continue
        statuses = metric_status(df, metric)
        for index, row in df.iterrows():
            rows.append({**{k: row[k] for k in ("answer_id", "question_id", "condition", "qtype")},
                "metric": metric, "value": row[metric], "status": statuses.loc[index],
                "reason": row.get(metric + "__reason", None)})
    return pd.DataFrame(rows)


def planned_contrasts(df, seed=42, bootstrap_samples=2000):
    rows = []
    for spec in analysis_plan()["contrasts"]:
        data = _subset(df, spec["subset"])
        a, b, metric = spec["condition_a"], spec["condition_b"], spec["metric"]
        data = data[data.condition.isin([a, b])]
        expected = int(data.question_id.nunique())
        if metric in data.columns:
            wide = data.pivot(index="question_id", columns="condition", values=metric).reindex(columns=[a, b])
            numeric = wide.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        else:
            numeric = pd.DataFrame(columns=[a, b])
        result = paired_comparison(numeric[a], numeric[b], spec["binary"], seed, bootstrap_samples)
        excluded = data[~data.question_id.isin(numeric.index)].copy()
        excluded["_metric_status"] = metric_status(excluded, metric)
        reason_counts = {}
        for name, statuses in {"evaluation_failure": ["evaluation_failed", "blocked"],
                               "not_applicable": ["not_applicable"], "pending": ["pending"],
                               "generation_failure": ["generation_failed"],
                               "other_missing": ["unknown_missing", "legacy_unversioned", "invalid_evaluation"]}.items():
            reason_counts["n_excluded_questions_with_" + name] = int(excluded.loc[
                excluded._metric_status.isin(statuses), "question_id"].nunique())
        rows.append({**spec, **result, "n_candidate_questions": expected, "n_missing_or_inapplicable": expected - len(numeric),
                     **reason_counts, "exclusion_counts_note": "Reason groups may overlap within a paired question"})
    for family in {r["family"] for r in rows}:
        members = [r for r in rows if r["family"] == family]
        for row, value in zip(members, holm_adjust([r["p"] for r in members])):
            row["p_holm"] = value
            row["family_size"] = len(members)
    return rows


def descriptives(df, groups):
    records = []
    metrics = [m for m in (*CONTINUOUS, *BINARY) if m in df.columns]
    for key, group in df.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(groups, key))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            records.append({**identity, "metric": metric, "n_cells": len(group), "n_applicable": len(values),
                            "n_observed": len(values), **missing_counts(metric_status(group, metric)),
                            "n_missing_or_inapplicable": len(group) - len(values), "mean": values.mean(),
                            "median": values.median(), "sd": values.std(ddof=1) if len(values) > 1 else None,
                            "min": values.min(), "max": values.max()})
    return pd.DataFrame(records)


def coverage_table(df):
    records = []
    for (condition, qtype), group in df.groupby(["condition", "qtype"]):
        success = group[group.status == "success"]
        counts = success.action.value_counts()
        row = {"condition": condition, "qtype": qtype, "expected_cells": len(group),
               "successful_cells": len(success), "technical_errors": int((group.status == "error").sum()),
               "not_attempted": int((group.status == "not_attempted").sum()),
               "answer_count": int(counts.get("ANSWER", 0)), "clarify_count": int(counts.get("CLARIFY", 0)),
               "refuse_count": int(counts.get("REFUSE", 0)),
               "response_coverage_including_technical_failures": float(counts.get("ANSWER", 0) / len(group))}
        if "evaluation_status" in group:
            for label in ("complete", "partial", "failed", "pending", "not_started", "legacy_unversioned", "invalid"):
                row["evaluation_" + label] = int((group.evaluation_status == label).sum())
            row["evaluation_incomplete_rate_among_successful_answers"] = (
                float((success.evaluation_status != "complete").mean()) if len(success) else None)
        for metric in BINARY:
            if metric in group:
                vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                row[metric + "_n"] = len(vals)
                row[metric + "_rate"] = vals.mean()
        records.append(row)
    return pd.DataFrame(records)


def agreement_pair(auto, human, binary=False):
    auto, human = np.asarray(auto, dtype=float), np.asarray(human, dtype=float)
    if len(auto) != len(human):
        raise ValueError("Agreement pairs must be aligned")
    n = len(auto)
    result = {"n": n}
    if not n:
        return result
    result.update(mae=float(np.abs(auto-human).mean()), bias_auto_minus_human=float((auto-human).mean()))
    if binary:
        if not np.isin(auto, [0, 1]).all() or not np.isin(human, [0, 1]).all():
            raise ValueError("Binary agreement received nonbinary labels")
        po = float((auto == human).mean())
        pe = sum(float((auto == v).mean() * (human == v).mean()) for v in (0, 1))
        result.update(raw_agreement=po, cohen_kappa=(po-pe)/(1-pe) if pe < 1 else None,
            confusion={f"human_{int(h)}_auto_{int(a)}": int(((human == h) & (auto == a)).sum()) for h in (0, 1) for a in (0, 1)})
        if pe == 1:
            result["note"] = "Kappa undefined: both label distributions are constant"
    else:
        result["spearman"] = float(stats.spearmanr(auto, human).statistic) if n >= 3 and len(set(auto)) > 1 and len(set(human)) > 1 else None
    return result


def auto_manual_agreement(df, manual_df):
    if manual_df is None or manual_df.empty:
        return {"status": "not_available", "note": "No independent human scores; RQ2 alignment is not established", "comparisons": []}
    manual = manual_df.copy()
    if "source" not in manual or not (manual.source == "human").any():
        return {"status": "not_available", "note": "AI-assisted ratings are not independent human validation", "comparisons": []}
    manual = manual[manual.source == "human"]
    if manual.duplicated(["answer_id", "evaluator"]).any():
        raise ValueError("Duplicate answer/evaluator ratings")
    matched = manual.merge(df.dropna(subset=["answer_id"]), on="answer_id", how="inner", suffixes=("_manual", "_auto"), validate="many_to_one")
    metrics = [m for m in (*CONTINUOUS[:7], "hallucinated", "action_appropriate", "appropriate_clarification")
               if m + "_manual" in matched and m + "_auto" in matched]
    comparisons = []
    for evaluator, rater_rows in matched.groupby("evaluator"):
        groups = [("overall", "all", rater_rows)]
        groups += [("condition", name, g) for name, g in rater_rows.groupby("condition")]
        groups += [("qtype", name, g) for name, g in rater_rows.groupby("qtype")]
        for dimension, label, group in groups:
            for metric in metrics:
                values = group[[metric + "_auto", metric + "_manual"]].apply(pd.to_numeric, errors="coerce").dropna()
                result = agreement_pair(values.iloc[:, 0], values.iloc[:, 1], metric in BINARY)
                comparisons.append({"evaluator": evaluator, "dimension": dimension, "group": label, "metric": metric,
                                    "n_questions": int(group.loc[values.index, "question_id"].nunique()), **result})
    if not any(result["n"] for result in comparisons):
        return {"status": "not_available", "comparisons": comparisons,
                "note": "No matched applicable numeric human scores; comments alone do not establish agreement"}
    return {"status": "available", "comparisons": comparisons,
            "n_evaluators": int(manual.evaluator.nunique()),
            "note": "Descriptive associations only; conditions share questions, so no independent-row correlation p-values. Single-rater results are not inter-rater reliability. Optional ordinal rubric is not silently equated with a continuous metric."}


def analyze_scores(df, output_dir, manual_df=None, seed=42, bootstrap_samples=2000):
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap draws (formal protocol defaults to 2000)")
    required = {"question_id", "condition", "qtype", "expected_action", "split", "status", "action", "answer_id"}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError("Missing analysis identifiers or empty dataset")
    if df.duplicated(["question_id", "condition"]).any():
        raise ValueError("Repeated question-condition cells cannot be treated as independent observations")
    if df.split.nunique() != 1:
        raise ValueError("Do not pool development and held-out test scores")
    # A non-ok metric never enters statistics, even if an imported frame contains a stale value.
    df = df.copy()
    for metric in (*CONTINUOUS, *BINARY):
        if metric in df and metric + "__status" in df:
            df[metric] = pd.to_numeric(df[metric], errors="coerce").astype(float).where(metric_status(df, metric) == "ok")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparisons = planned_contrasts(df, seed, bootstrap_samples)
    agreement = auto_manual_agreement(df, manual_df)
    df.to_csv(out / "scores.csv", index=False)
    descriptives(df, ["condition"]).to_csv(out / "descriptives.csv", index=False)
    descriptives(df, ["condition", "qtype"]).to_csv(out / "by_question_type.csv", index=False)
    coverage_table(df).to_csv(out / "coverage_and_actions.csv", index=False)
    metric_availability(df).to_csv(out / "metric_availability.csv", index=False)
    if "evaluation_status" in df:
        df[(df.status == "success") & (df.evaluation_status != "complete")].to_csv(
            out / "evaluation_failures.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_contrasts.csv", index=False)
    df[df.status != "success"].to_csv(out / "technical_failures.csv", index=False)
    diagnostic_cols = [c for c in df if c.startswith("diagnostic_")]
    if diagnostic_cols:
        df.groupby(["condition", "qtype"])[diagnostic_cols].agg(["mean", "count", "sum"]).to_csv(out / "failure_diagnostics.csv")
    summary = {"split": str(df.split.iloc[0]), "n_questions": int(df.question_id.nunique()),
               "n_result_cells": len(df), "n_successful": int((df.status == "success").sum()),
               "plan": analysis_plan(), "bootstrap_samples_used": bootstrap_samples, "seed": seed,
               "contrasts": comparisons,
               "caveats": ["Quality uses joint applicable responses, not unconditional performance.",
                           "Partial evaluation preserves validated metrics only. Failed/blocked metrics are missing, never zero or genuine N/A; exclusions may be selective.",
                           "Confidence intervals are paired-question percentile bootstrap intervals, not multiplicity-adjusted intervals.",
                           "No significance does not establish equivalence or noninferiority.",
                           "All questions must be independently sampled units; semantic families must not be disguised as separate questions."]}
    (out / "tests.json").write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    (out / "auto_manual_agreement.json").write_text(json.dumps(clean_json(agreement), indent=2, allow_nan=False), encoding="utf-8")
    if manual_df is not None and not manual_df.empty and "error_category" in manual_df:
        manual_df.groupby(["evaluator", "error_category"], dropna=False).size().rename("count").to_csv(out / "manual_error_categories.csv")
    if manual_df is not None and not manual_df.empty and {"required_fact_coverage", "source"}.issubset(manual_df.columns):
        coverage = manual_df.loc[manual_df.source == "human", ["answer_id", "evaluator", "required_fact_coverage"]].copy()
        if coverage.duplicated(["answer_id", "evaluator"]).any():
            raise ValueError("Duplicate manual required-fact ratings")
        coverage = coverage.merge(df[["answer_id", "question_id", "condition", "qtype"]].dropna(subset=["answer_id"]),
                                  on="answer_id", how="inner", validate="many_to_one")
        coverage["required_fact_coverage"] = pd.to_numeric(coverage.required_fact_coverage, errors="raise")
        observed = coverage.required_fact_coverage.dropna()
        if not observed.between(0, 1).all():
            raise ValueError("Manual required_fact_coverage must be within 0..1")
        coverage.to_csv(out / "manual_required_fact_coverage.csv", index=False)
        coverage.groupby(["evaluator", "condition", "qtype"]).required_fact_coverage.agg(["count", "mean"]).to_csv(
            out / "manual_required_fact_coverage_summary.csv")
    return {"out": str(out), "n_questions": summary["n_questions"], "split": summary["split"],
            "contrasts": len(comparisons), "human_agreement": agreement["status"]}
