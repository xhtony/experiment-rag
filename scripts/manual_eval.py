"""Manual rubric evaluation workflow.

1) Export answers (blinded: condition hidden, randomized order) to CSV:
       python scripts/manual_eval.py export --run-id 1 --out manual_run1.csv
2) The evaluator fills in the rubric columns:
       faithfulness_rubric: 2 = fully supported, 1 = partial, 0 = unsupported
       citation_correct:    1/0
       hallucinated:        1/0
       refusal_correct:     1/0 (only for refusals / unanswerable questions)
       error_category:      retrieval_failure | ranking_failure | generation_failure |
                            citation_failure | over_answering | over_refusal |
                            policy_interpretation_error | none
3) Import the completed sheet:
       python scripts/manual_eval.py import --file manual_run1.csv --evaluator alice
"""

import argparse
import random

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from ragapp.storage.db import session_scope
from ragapp.storage.models import Answer, Chunk, ManualScore, Question


def export(run_id: int, out: str, sample: int | None) -> None:
    with session_scope() as session:
        rows = (
            session.query(Answer, Question)
            .join(Question, Question.id == Answer.question_id)
            .filter(Answer.run_id == run_id)
            .all()
        )
        records = []
        for ans, q in rows:
            ctx_ids = ans.retrieved_chunk_ids or q.gold_chunk_ids or []
            chunks = session.query(Chunk).filter(Chunk.id.in_(ctx_ids)).all() if ctx_ids else []
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            records.append(
                {
                    "answer_id": ans.id,
                    "question": q.text,
                    "qtype": q.qtype,
                    "gold_answer": q.gold_answer,
                    "context": context,
                    "answer_text": ans.answer_text,
                    "refused": ans.refused,
                    "refusal_reason": ans.refusal_reason,
                    "citations": ", ".join(ans.cited_chunk_ids or []),
                    # rubric columns to fill in:
                    "faithfulness_rubric": "",
                    "citation_correct": "",
                    "hallucinated": "",
                    "refusal_correct": "",
                    "error_category": "",
                    "notes": "",
                }
            )
    random.Random(42).shuffle(records)  # blind + fixed order across evaluators
    if sample:
        records = records[:sample]
    pd.DataFrame(records).to_csv(out, index=False)
    print(f"Exported {len(records)} answers to {out} (condition hidden, order randomized).")


def import_scores(file: str, evaluator: str) -> None:
    df = pd.read_csv(file)
    n = 0
    with session_scope() as session:
        for _, row in df.iterrows():
            if pd.isna(row.get("faithfulness_rubric")) or row.get("faithfulness_rubric") == "":
                continue

            def as_bool(v):
                return None if pd.isna(v) or v == "" else bool(int(v))

            answer_id = int(row["answer_id"])
            score = (
                session.query(ManualScore)
                .filter_by(answer_id=answer_id, evaluator=evaluator)
                .one_or_none()
            ) or ManualScore(answer_id=answer_id, evaluator=evaluator)
            score.faithfulness_rubric = int(row["faithfulness_rubric"])
            score.citation_correct = as_bool(row.get("citation_correct"))
            score.hallucinated = as_bool(row.get("hallucinated"))
            score.refusal_correct = as_bool(row.get("refusal_correct"))
            score.error_category = str(row.get("error_category") or "")
            score.notes = str(row.get("notes") or "")
            session.add(score)
            n += 1
    print(f"Imported {n} manual scores from {file} (evaluator={evaluator}).")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export")
    p_exp.add_argument("--run-id", type=int, required=True)
    p_exp.add_argument("--out", required=True)
    p_exp.add_argument("--sample", type=int, default=None,
                       help="Export only a random sample of N answers.")

    p_imp = sub.add_parser("import")
    p_imp.add_argument("--file", required=True)
    p_imp.add_argument("--evaluator", required=True)

    args = parser.parse_args()
    if args.cmd == "export":
        export(args.run_id, args.out, args.sample)
    else:
        import_scores(args.file, args.evaluator)


if __name__ == "__main__":
    main()
