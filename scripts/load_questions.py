"""Load the gold-standard question dataset (JSONL) into PostgreSQL.

Each line:
{
  "id": 1,
  "question": "...",
  "qtype": "single_doc|multi_doc|conditional|ambiguous|unanswerable",
  "is_answerable": true,
  "expect_refusal": false,
  "gold_answer": "...",
  "gold_chunk_ids": ["late-submission-policy::0001"]
}

Usage:
    python scripts/load_questions.py --file data/questions/questions.jsonl [--replace]
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ragapp.storage.db import init_db, session_scope
from ragapp.storage.models import Question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/questions/questions.jsonl")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    init_db()
    rows = [
        json.loads(line)
        for line in Path(args.file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with session_scope() as session:
        if args.replace:
            session.query(Question).delete()
        for r in rows:
            session.merge(
                Question(
                    id=r["id"],
                    text=r["question"],
                    qtype=r["qtype"],
                    is_answerable=r.get("is_answerable", True),
                    expect_refusal=r.get("expect_refusal", False),
                    gold_answer=r.get("gold_answer", ""),
                    gold_chunk_ids=r.get("gold_chunk_ids", []),
                )
            )
    print(f"Loaded {len(rows)} questions.")


if __name__ == "__main__":
    main()
