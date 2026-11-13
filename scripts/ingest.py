"""Ingest the controlled corpus into PostgreSQL + Qdrant.

Usage:
    python scripts/ingest.py --corpus data/corpus [--keep] [--show-chunks]
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ragapp.shared.corpus import ingest
from ragapp.storage.db import init_db, session_scope
from ragapp.storage.models import Chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--keep", action="store_true",
                        help="Do not wipe existing index before ingesting.")
    parser.add_argument("--show-chunks", action="store_true",
                        help="Print all chunk ids (useful when labelling gold chunks).")
    args = parser.parse_args()

    init_db()
    stats = ingest(Path(args.corpus), recreate=not args.keep)
    print(f"Ingested {stats['documents']} documents, {stats['chunks']} chunks.")

    if args.show_chunks:
        with session_scope() as session:
            for c in session.query(Chunk).order_by(Chunk.id).all():
                preview = " ".join(c.text.split())[:90]
                print(f"{c.id} | {c.section} | {preview}")


if __name__ == "__main__":
    main()
