"""Corpus ingestion: load documents, chunk with provenance metadata, and write
to both PostgreSQL (source of truth, BM25) and Qdrant (dense index).

Supported formats:
- Markdown / text files with optional YAML front matter:
    ---
    title: Late Submission Policy
    version: "2025-1"
    date: "2025-02-01"
    url: https://example.ac.nz/policy/late-submission
    ---
- PDF files (page numbers recorded per chunk).

Chunk ids are stable and human-readable: "<doc-slug>::<seq:04d>", so gold
labels and citations remain valid across re-runs as long as the corpus and
chunking parameters are unchanged.
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragapp.config import get_settings
from ragapp.shared.vectorstore import ensure_collection, get_vectorstore
from ragapp.storage.db import session_scope
from ragapp.storage.models import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class ParsedDoc:
    slug: str
    title: str
    version: str = ""
    date: str = ""
    url: str = ""
    source_path: str = ""
    # list of (section_heading, text, page)
    sections: list[tuple[str, str, int | None]] = field(default_factory=list)


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _parse_markdown(path: Path) -> ParsedDoc:
    post = frontmatter.load(str(path))
    meta = post.metadata
    title = str(meta.get("title", path.stem))
    doc = ParsedDoc(
        slug=_slugify(path.stem),
        title=title,
        version=str(meta.get("version", "")),
        date=str(meta.get("date", "")),
        url=str(meta.get("url", "")),
        source_path=str(path),
    )
    current_heading, buf = "", []
    for line in post.content.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buf and "".join(buf).strip():
                doc.sections.append((current_heading, "\n".join(buf).strip(), None))
            current_heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf and "".join(buf).strip():
        doc.sections.append((current_heading, "\n".join(buf).strip(), None))
    return doc


def _parse_pdf(path: Path) -> ParsedDoc:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    doc = ParsedDoc(slug=_slugify(path.stem), title=path.stem, source_path=str(path))
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            doc.sections.append(("", text, page_no))
    return doc


def load_corpus(corpus_dir: Path) -> list[ParsedDoc]:
    docs = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            docs.append(_parse_markdown(path))
        elif path.suffix.lower() == ".pdf":
            docs.append(_parse_pdf(path))
    return docs


def ingest(corpus_dir: Path, recreate: bool = True) -> dict:
    """Chunk all documents and index them in PostgreSQL + Qdrant."""
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap
    )
    parsed = load_corpus(corpus_dir)
    if not parsed:
        raise RuntimeError(f"No documents found in {corpus_dir}")

    ensure_collection(recreate=recreate)
    lc_docs: list[LCDocument] = []
    ids: list[str] = []

    with session_scope() as session:
        if recreate:
            session.query(Chunk).delete()
            session.query(Document).delete()

        for doc in parsed:
            session.merge(
                Document(
                    id=doc.slug,
                    title=doc.title,
                    version=doc.version,
                    doc_date=doc.date,
                    source_path=doc.source_path,
                    source_url=doc.url,
                )
            )
            seq = 0
            for section, text, page in doc.sections:
                for piece in splitter.split_text(text):
                    chunk_id = f"{doc.slug}::{seq:04d}"
                    session.add(
                        Chunk(
                            id=chunk_id,
                            document_id=doc.slug,
                            seq=seq,
                            section=section,
                            page=page,
                            text=piece,
                        )
                    )
                    lc_docs.append(
                        LCDocument(
                            page_content=piece,
                            metadata={
                                "chunk_id": chunk_id,
                                "document_title": doc.title,
                                "section": section,
                                "page": page,
                                "version": doc.version,
                            },
                        )
                    )
                    # Qdrant requires UUID/int point ids; derive deterministically
                    ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)))
                    seq += 1

    get_vectorstore().add_documents(lc_docs, ids=ids)
    return {"documents": len(parsed), "chunks": len(lc_docs)}
