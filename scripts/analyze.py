"""Statistical analysis of a fully evaluated run.

Produces:
- results/run<id>_scores.csv          raw per-answer metric table
- results/run<id>_descriptives.csv    mean/median/SD per condition
- results/run<id>_refusal.csv         refusal confusion metrics per condition
- results/run<id>_tests.json          Friedman / Wilcoxon+Holm / effect sizes
- console summary mapped to hypotheses H1-H5

Usage:
    python scripts/analyze.py --run-id 1 [--out results]
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ragapp.analysis import stats as st


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = st.load_scores(args.run_id)
    if df.empty:
        print("No scores found - run evaluate.py first.")
        return
    df.to_csv(out / f"run{args.run_id}_scores.csv", index=False)

    # --- descriptives ---
    desc = (
        df.groupby("condition")[
            ["faithfulness", "citation_precision", "citation_recall",
             "citation_f1", "citation_support", "hallucinated", "retrieval_recall",
             "latency_ms"]
        ]
        .agg(["mean", "median", "std", "count"])
        .round(4)
    )
    desc.to_csv(out / f"run{args.run_id}_descriptives.csv")
    print("=== Descriptive statistics (per condition) ===")
    print(desc.to_string())

    # --- refusal behaviour ---
    ref = st.refusal_metrics(df)
    ref.to_csv(out / f"run{args.run_id}_refusal.csv", index=False)
    print("\n=== Refusal behaviour ===")
    print(ref.round(4).to_string(index=False))

    unanswerable = df[df.expect_refusal]
    if not unanswerable.empty:
        print("\n=== Refusal behaviour (unanswerable subset, H4) ===")
        print(st.refusal_metrics(unanswerable).round(4).to_string(index=False))

    # --- inferential tests ---
    tests = {m: st.compare_conditions(df, m) for m in st.METRICS}
    (out / f"run{args.run_id}_tests.json").write_text(
        json.dumps(tests, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== Hypothesis tests ===")
    for metric, res in tests.items():
        if "pairwise" not in res:
            continue
        fr = res.get("friedman")
        head = f"[{metric}] n={res['n_questions']}"
        if fr:
            head += f" | Friedman chi2={fr['statistic']:.3f}, p={fr['p']:.4g}"
        if "kendalls_w" in res:
            head += f", Kendall W={res['kendalls_w']:.3f}"
        print(head)
        for pw in res["pairwise"]:
            print(
                f"   {pw['pair']}: mean {pw.get('mean_a', float('nan')):.3f} vs "
                f"{pw.get('mean_b', float('nan')):.3f}, "
                f"Wilcoxon p={pw['wilcoxon_p']:.4g}, Holm p={pw['p_holm']:.4g}, "
                f"Cliff's delta={pw['cliffs_delta']:.3f}"
            )

    # --- H5: automated vs manual ---
    manual = st.load_manual_scores(args.run_id)
    agree = st.auto_vs_manual(df, manual)
    print("\n=== H5: automated vs manual agreement ===")
    print(json.dumps(agree, indent=2, default=str))

    print(f"\nArtifacts written to {out}/")


if __name__ == "__main__":
    main()
