"""Statistical analysis for the repeated-measures design.

Implements the analysis plan from the methodology:
- descriptive statistics per condition
- Shapiro-Wilk normality check
- omnibus: Friedman test (non-parametric repeated measures)
- pairwise: Wilcoxon signed-rank with Holm-Bonferroni correction
- effect sizes: Cliff's delta, Kendall's W
- H5: Spearman/Pearson correlation + Cohen's kappa / ICC between automated
  and manual scores
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats as ss

from ragapp.storage.db import session_scope
from ragapp.storage.models import Answer, AutoScore, ManualScore, Question

METRICS = [
    "faithfulness",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "citation_support",
    "hallucinated",
    "retrieval_recall",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_scores(run_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = (
            session.query(Answer, AutoScore, Question)
            .join(AutoScore, AutoScore.answer_id == Answer.id)
            .join(Question, Question.id == Answer.question_id)
            .filter(Answer.run_id == run_id)
            .all()
        )
        records = []
        for ans, sc, q in rows:
            records.append(
                {
                    "answer_id": ans.id,
                    "question_id": ans.question_id,
                    "condition": ans.condition,
                    "qtype": q.qtype,
                    "expect_refusal": q.expect_refusal,
                    "refused": ans.refused,
                    "latency_ms": ans.latency_ms,
                    "faithfulness": sc.faithfulness,
                    "citation_precision": sc.citation_precision,
                    "citation_recall": sc.citation_recall,
                    "citation_f1": sc.citation_f1,
                    "citation_support": sc.citation_support,
                    "hallucinated": None if sc.hallucinated is None else float(sc.hallucinated),
                    "refusal_outcome": sc.refusal_outcome,
                    "retrieval_recall": sc.retrieval_recall,
                }
            )
    return pd.DataFrame(records)


def load_manual_scores(run_id: int) -> pd.DataFrame:
    with session_scope() as session:
        rows = (
            session.query(ManualScore, Answer)
            .join(Answer, Answer.id == ManualScore.answer_id)
            .filter(Answer.run_id == run_id)
            .all()
        )
        records = [
            {
                "answer_id": ans.id,
                "condition": ans.condition,
                "question_id": ans.question_id,
                "evaluator": ms.evaluator,
                "manual_faithfulness": ms.faithfulness_rubric,
                "manual_citation_correct": ms.citation_correct,
                "manual_hallucinated": ms.hallucinated,
                "manual_refusal_correct": ms.refusal_correct,
                "error_category": ms.error_category,
            }
            for ms, ans in rows
        ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))


def _tie_correction(ranks_row: np.ndarray) -> float:
    """T_j = sum (t^3 - t) over tied-rank groups in one subject's ranking."""
    _, counts = np.unique(ranks_row, return_counts=True)
    return float(np.sum(counts**3 - counts))


def kendalls_w(matrix: np.ndarray) -> float:
    """Kendall's coefficient of concordance with tie correction.

    ``matrix`` is n_subjects × k_conditions. Each subject ranks the k
    conditions (average ranks for ties via ``rankdata``). Uses the
    Siegel & Castellan correction:

        W = 12 S / (n² (k³ - k) - n Σ T_j)

    where T_j = Σ (t_g³ - t_g) for tied groups in subject j. Without ties
    this reduces to the uncorrected form 12 S / (n² (k³ - k)).
    """
    n, k = matrix.shape
    ranks = np.apply_along_axis(ss.rankdata, 1, matrix)
    r_sum = ranks.sum(axis=0)
    s = ((r_sum - r_sum.mean()) ** 2).sum()
    t_sum = sum(_tie_correction(row) for row in ranks)
    denom = n**2 * (k**3 - k) - n * t_sum
    if denom <= 0:
        return 0.0
    return float(12 * s / denom)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Per-metric repeated-measures comparison
# ---------------------------------------------------------------------------

def compare_conditions(df: pd.DataFrame, metric: str) -> dict:
    """Friedman omnibus + pairwise Wilcoxon with Holm correction on the subset
    of questions answered (and scored) under every condition."""
    wide = df.pivot_table(index="question_id", columns="condition", values=metric)
    wide = wide.dropna()
    result: dict = {"metric": metric, "n_questions": len(wide), "conditions": list(wide.columns)}
    if len(wide) < 5 or wide.shape[1] < 2:
        result["note"] = "insufficient paired data"
        return result

    # Normality (reported, but inference uses non-parametric tests)
    result["shapiro"] = {
        c: round(float(ss.shapiro(wide[c]).pvalue), 4)
        if wide[c].nunique() > 1 else None
        for c in wide.columns
    }

    if wide.shape[1] >= 3:
        stat, p = ss.friedmanchisquare(*[wide[c] for c in wide.columns])
        result["friedman"] = {"statistic": float(stat), "p": float(p)}
        result["kendalls_w"] = float(kendalls_w(wide.to_numpy()))

    pairs = list(itertools.combinations(wide.columns, 2))
    raw_p, pair_stats = [], []
    for a, b in pairs:
        diff = wide[a] - wide[b]
        if (diff == 0).all():
            raw_p.append(1.0)
            pair_stats.append({"pair": f"{a} vs {b}", "wilcoxon_p": 1.0, "cliffs_delta": 0.0})
            continue
        w = ss.wilcoxon(wide[a], wide[b], zero_method="wilcox")
        raw_p.append(float(w.pvalue))
        pair_stats.append(
            {
                "pair": f"{a} vs {b}",
                "mean_a": float(wide[a].mean()),
                "mean_b": float(wide[b].mean()),
                "wilcoxon_p": float(w.pvalue),
                "cliffs_delta": float(cliffs_delta(wide[a].to_numpy(), wide[b].to_numpy())),
            }
        )
    adj = holm_bonferroni(raw_p)
    for st, p_adj in zip(pair_stats, adj):
        st["p_holm"] = float(p_adj)
    result["pairwise"] = pair_stats
    return result


# ---------------------------------------------------------------------------
# Refusal behaviour metrics (per condition)
# ---------------------------------------------------------------------------

def refusal_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition"):
        tp = (g.refusal_outcome == "correct_refusal").sum()   # refused & should refuse
        fp = (g.refusal_outcome == "false_refusal").sum()     # refused & answerable
        fn = (g.refusal_outcome == "false_answer").sum()      # answered & should refuse
        tn = (g.refusal_outcome == "correct_answer").sum()
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and not np.isnan(precision) and not np.isnan(recall)
            else np.nan
        )
        rows.append(
            {
                "condition": cond,
                "n": len(g),
                "correct_refusal": tp,
                "false_refusal": fp,
                "false_answer": fn,
                "correct_answer": tn,
                "refusal_accuracy": (tp + tn) / len(g),
                "refusal_precision": precision,
                "refusal_recall": recall,
                "refusal_f1": f1,
                "false_refusal_rate": fp / (fp + tn) if (fp + tn) else np.nan,
                "false_answer_rate": fn / (fn + tp) if (fn + tp) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("condition")


# ---------------------------------------------------------------------------
# H5: automated vs manual agreement
# ---------------------------------------------------------------------------

def auto_vs_manual(auto_df: pd.DataFrame, manual_df: pd.DataFrame) -> dict:
    if manual_df.empty:
        return {"note": "no manual scores available"}
    merged = manual_df.merge(
        auto_df[["answer_id", "faithfulness", "hallucinated"]], on="answer_id"
    )
    out: dict = {"n_pairs": len(merged)}

    pair = merged.dropna(subset=["manual_faithfulness", "faithfulness"])
    if len(pair) >= 5:
        # manual rubric 0..2 -> 0..1 for comparability
        manual = pair["manual_faithfulness"].astype(float) / 2.0
        auto = pair["faithfulness"].astype(float)
        out["faithfulness_spearman"] = tuple(map(float, ss.spearmanr(manual, auto)))
        out["faithfulness_pearson"] = tuple(map(float, ss.pearsonr(manual, auto)))

    pair_h = merged.dropna(subset=["manual_hallucinated", "hallucinated"])
    if len(pair_h) >= 5:
        try:
            from sklearn.metrics import cohen_kappa_score  # optional
            out["hallucination_kappa"] = float(
                cohen_kappa_score(
                    pair_h["manual_hallucinated"].astype(int),
                    pair_h["hallucinated"].astype(int),
                )
            )
        except ImportError:
            out["hallucination_kappa"] = _cohen_kappa(
                pair_h["manual_hallucinated"].astype(int).to_numpy(),
                pair_h["hallucinated"].astype(int).to_numpy(),
            )

    # Inter-rater reliability if >= 2 evaluators (ICC via pingouin)
    if manual_df["evaluator"].nunique() >= 2:
        try:
            import pingouin as pg

            icc = pg.intraclass_corr(
                data=manual_df.dropna(subset=["manual_faithfulness"]),
                targets="answer_id",
                raters="evaluator",
                ratings="manual_faithfulness",
            )
            out["icc"] = icc[icc["Type"] == "ICC2"].to_dict("records")
        except Exception as exc:  # pingouin optional at runtime
            out["icc_error"] = str(exc)
    return out


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = sum(((a == v).mean() * (b == v).mean()) for v in np.unique(np.concatenate([a, b])))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
