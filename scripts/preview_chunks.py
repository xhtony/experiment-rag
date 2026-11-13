"""Dry-run chunk preview WITHOUT PostgreSQL/Qdrant.

Replicates the exact parsing + chunking logic of ragapp.shared.corpus so that
gold_chunk_ids can be labelled before infrastructure is up. Chunk ids printed
here are identical to those produced by scripts/ingest.py (same corpus, same
chunking parameters).

Usage:
    python scripts/preview_chunks.py [--corpus data/corpus] [--chunk-size 800] [--overlap 120]
"""

import argparse
import re
from pathlib import Path

import frontmatter
from langchain_text_splitters import RecursiveCharacterTextSplitter

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_markdown(path: Path):
    post = frontmatter.load(str(path))
    sections = []
    current_heading, buf = "", []
    for line in post.content.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buf and "".join(buf).strip():
                sections.append((current_heading, "\n".join(buf).strip()))
            current_heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        sections.append((current_heading, "\n".join(buf).strip()))
    return _slugify(path.stem), sections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.overlap
    )
    total = 0
    for path in sorted(Path(args.corpus).rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        slug, sections = parse_markdown(path)
        seq = 0
        for section, text in sections:
            pieces = splitter.split_text(text)
            for piece in pieces:
                chunk_id = f"{slug}::{seq:04d}"
                preview = " ".join(piece.split())[:80]
                multi = f" [section split into {len(pieces)}]" if len(pieces) > 1 else ""
                print(f"{chunk_id} | {len(piece):4d} ch | {section}{multi} | {preview}")
                seq += 1
                total += 1
    print(f"\nTotal chunks: {total}")


if __name__ == "__main__":
    main()
